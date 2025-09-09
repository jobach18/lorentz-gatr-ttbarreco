import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

import os, time
import json
from omegaconf import open_dict

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

from experiments.base_experiment import BaseExperiment
from experiments.top_reco.dataset import TopRecoDataset
from experiments.top_reco.plots import plot_mixer
from experiments.top_reco.embedding import embed_tagging_data_into_ga
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow

MODEL_TITLE_DICT = {"GATr": "GATr"}

UNITS = 40  # We use units of 40 GeV for all tagging experiments

def mse_symmetrized(predictions: torch.Tensor,
             targets: torch.Tensor,
             *,
             allow_swap: bool = True,
             ) -> torch.Tensor:
    """Calculate the mean squared error loss.

    Args:
        predictions (torch.Tensor): The predicted values. Shape (batch_size, 2, 4).
        targets (torch.Tensor): The ground truth values. Shape (batch_size, 2, 4).

    Returns:
        torch.Tensor: The computed mean squared error loss.

    """
    predictions = predictions[0,:,:,:]
    targets = targets[0,:,:,:]
    #print(f' for debugging in the loss: predictions shape: {predictions.shape} targets shape: {targets.shape}')
    if not allow_swap:
        return F.mse_loss(predictions, targets, reduction="mean")

    # Take the minimum loss for each event
    # Option 1: no swap
    mse_noswap = F.mse_loss(predictions, targets, reduction="none").mean(dim=(1, 2))

    # Option 2: swap predicted tops
    pred_swapped = predictions[:, [1, 0], :]
    mse_swap = F.mse_loss(pred_swapped, targets, reduction="none").mean(dim=(1, 2))
    return torch.mean(torch.minimum(mse_noswap, mse_swap))

class RecoExperiment(BaseExperiment):
    """
    Base class for Top Reco 
    """

    def _init_loss(self):
        #self.loss = torch.nn.MSELoss()
        self.loss = mse_symmetrized
        #self.loss = torch.nn.SmoothL1Loss(beta=0.01)

    def init_physics(self):
        if not self.cfg.training.force_xformers:
            LOGGER.warning(
                f"Using training.force_xformers=False, this will slow down the network by a factor of 5-10."
            )

        # dynamically extend dict
        with open_dict(self.cfg):
            # extra mv channels for beam_reference and time_reference
            if not self.cfg.data.beam_token:
                if self.cfg.data.beam_reference is not None:
                    self.cfg.model.net.in_mv_channels += (
                        2
                        if self.cfg.data.two_beams
                        and not self.cfg.data.beam_reference == "xyplane"
                        else 1
                    )
                if self.cfg.data.add_time_reference:
                    self.cfg.model.net.in_mv_channels += 1

            # reinsert channels
            if self.cfg.data.reinsert_channels:
                self.cfg.model.net.reinsert_mv_channels = list(
                    range(self.cfg.model.net.in_mv_channels)
                )
                self.cfg.model.net.reinsert_s_channels = list(
                    range(self.cfg.model.net.in_s_channels)
                )

    def init_data(self):
        raise NotImplementedError

    def _init_data(self, Dataset, data_path, data_path_val):
        LOGGER.info(f"Creating {Dataset.__name__} from {data_path}")
        t0 = time.time()
        self.data_train = Dataset()
        self.data_val = Dataset()
        self.data_train.load_data(data_path, scalar_target=self.cfg.training.scalar_target)#, data_scale='std')
        self.data_val.load_data(data_path_val, scalar_target=self.cfg.training.scalar_target)#, data_scale='std')
        dt = time.time() - t0
        LOGGER.info(f"Finished creating datasets after {dt:.2f} s = {dt/60:.2f} min")

    def _init_dataloader(self):
        self.train_loader = DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.training.batchsize,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )

        LOGGER.info(
            f"Constructed dataloaders with "
            f"train_batches={len(self.train_loader)}, val_batches={len(self.val_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), {self.cfg.evaluation.batchsize} (evaluation)"
        )
        for batch in self.train_loader:
            data = batch.x
            #print(f'the input data is of size: {data.shape}')
            labels = batch.targets
            #print(f'the targets during loading are {batch.targets.shape}')
            scalars_targ = batch.scalar_targets if hasattr(batch, 'scalar_targets') else None
            if scalars_targ is not None:
                #print(f'the scalars during loading are {scalars.shape}')
                assert scalars_targ.shape[0] == data.shape[0], "Scalars and data must match size"
            break

    def evaluate(self):
        self.results = {}
        loader_dict = {
            "train": self.train_loader,
            "val": self.val_loader,
        }
        for set_label in self.cfg.evaluation.eval_set:
            if self.ema is not None:
                with self.ema.average_parameters():
                    self.results[set_label] = self._evaluate_single(
                        loader_dict[set_label], set_label, mode="eval"
                    )

                self._evaluate_single(
                    loader_dict[set_label], f"{set_label}_noema", mode="eval"
                )

            else:
                # self.results[set_label] = self._evaluate_single(
                #     loader_dict[set_label], set_label
                # )               
                
                result = self._evaluate_single(loader_dict[set_label], set_label)
                
                result_temp = result
                result_temp["val"]["raw"]["truth"] = result_temp["val"]["raw"]["truth"].tolist()
                result_temp["val"]["raw"]["input"] = result_temp["val"]["raw"]["input"].tolist()
                result_temp["val"]["raw"]["prediction"] = result_temp["val"]["raw"]["prediction"].tolist()
                if result_temp["val"]["raw"]["scalar_truth"] is not None:
                    result_temp["val"]["raw"]["scalar_truth"] = result_temp["val"]["raw"]["scalar_truth"].tolist()
                if result_temp["val"]["raw"]["scalar_prediction"] is not None:
                    result_temp["val"]["raw"]["scalar_prediction"] = result_temp["val"]["raw"]["scalar_prediction"].tolist()
                result_temp["val"]["raw"]["true_pt"] = result_temp["val"]["raw"]["true_pt"].tolist()
                result_temp["val"]["raw"]["pred_pt"] = result_temp["val"]["raw"]["pred_pt"].tolist()
                
                LOGGER.info(f"Creating result_{set_label}.json.")
                os.makedirs("results_to_notebook/mse_symmetrized", exist_ok=True)
                json_path = os.path.join("results_to_notebook/mse_symmetrized", f"result_{set_label}.json")
                with open(json_path, "w") as json_file:
                    json.dump(result_temp, json_file, default=str)

                self.results[set_label] = result
                #self.results[set_label] = self._evaluate_single(
                #    loader_dict[set_label], set_label
                #)
            return result['val']['raw']['mse']

    def _evaluate_single(self, loader, title, step=None):
        # compute predictions
        # note: shuffle=True or False does not matter, because we take the predictions directly from the dataloader and not from the dataset
        amplitudes_truth_prepd = [[] for _ in range(1)]
        amplitudes_lepton_prepd = [[] for _ in range(1)]
        amplitudes_pred_prepd = [[] for _ in range(1)]
        amplitudes_scalar_truth_prepd = [[] for _ in range(1)]
        amplitudes_scalar_pred_prepd = [[] for _ in range(1)]
        LOGGER.info(f"### Starting to evaluate model on {title} dataset ###")
        self.model.eval()
        if self.cfg.training.optimizer == "ScheduleFree":
            self.optimizer.eval()
        t0 = time.time()
        for data in loader:
            data.to(self.device)
            y = data['targets'] 
            y_sc = data['scalar_targets'] if hasattr(data, 'scalar_targets') else None
            #print(f'the target shape is {y.shape}')
            embedding = embed_tagging_data_into_ga(
                data.x, data.scalars, data.ptr, self.cfg.data
            )
            pred, pred_sc = self.model(
                    embedding
            )
            #print(f'the network prediction during eval is {pred.cpu().float().detach().numpy().shape}')
            #print(f'the truth during eval is {y.cpu().float().detach().numpy().shape}')

            amplitudes_pred_prepd[0].append(pred.cpu().float().detach().numpy())
            amplitudes_lepton_prepd[0].append(data.x.cpu().float().detach().numpy())
            amplitudes_truth_prepd[0].append(
                y.cpu().float().detach().numpy().reshape((1,int(
                 y.shape[0] / 8 ), 2, 4,
                 ))
                )
            amplitudes_scalar_pred_prepd[0].append(
                pred_sc.cpu().float().detach().numpy(
                 )) if y_sc is not None else None
            amplitudes_scalar_truth_prepd[0].append(
                y_sc.cpu().float().detach().numpy().reshape((1,int(
                 y_sc.shape[0] / 2 ), 2,
                 )) if y_sc is not None else None
            )
            
        amplitudes_pred_prepd = [
                np.concatenate(individual) for individual in amplitudes_pred_prepd[0]
        ]
        amplitudes_truth_prepd = [
                np.concatenate(individual) for individual in amplitudes_truth_prepd[0]
        ]
        amplitudes_lepton_prepd = [
                np.concatenate(individual) for individual in amplitudes_lepton_prepd[0]
        ]
        amplitudes_scalar_pred_prepd = [
                np.concatenate(individual) for individual in amplitudes_scalar_pred_prepd[0]
        ] if y_sc is not None else None
        amplitudes_scalar_truth_prepd = [
                np.concatenate(individual) for individual in amplitudes_scalar_truth_prepd[0]
        ] if y_sc is not None else None 
        dt = (
            (time.time() - t0)
            * 1e6
            / sum(arr.shape[0] for arr in amplitudes_truth_prepd)
        )
        LOGGER.info(
            f"Evaluation time: {dt:.2f}s for 1M events "
            f"using batchsize {self.cfg.evaluation.batchsize}"
        )

        results = {}
        amp_pred = np.concatenate(amplitudes_pred_prepd, axis=0)
        amp_truth = np.concatenate(amplitudes_truth_prepd, axis=0)
        amp_lepton = np.concatenate(amplitudes_lepton_prepd, axis=0)
        if y_sc is not None:
            amplitudes_scalar_pred_prepd = np.concatenate(amplitudes_scalar_pred_prepd, axis=0)
            amplitudes_scalar_truth_prepd = np.concatenate(amplitudes_scalar_truth_prepd, axis=0)
            LOGGER.info(f'the scalar pred shape is {amplitudes_scalar_pred_prepd.shape}')
            LOGGER.info(f'the scalar truth shape is {amplitudes_scalar_truth_prepd.shape}')
        else:
            amplitudes_scalar_pred_prepd = None
            amplitudes_scalar_truth_prepd = None
        #amp_pred = amplitudes_pred_prepd[0]
        #amp_truth = amplitudes_truth_prepd[0]


        pred_pt = np.sqrt(np.sum(amp_pred[:,0,1:2]**2, axis=-1)) + np.sqrt(np.sum(amp_pred[:,1,1:2]**2, axis=-1) )
        true_pt = np.sqrt(np.sum(amp_truth[:,0,1:2]**2, axis=-1)) + np.sqrt(np.sum(amp_truth[:,1,1:2]**2, axis=-1) )
        LOGGER.info(f'the true pt shape is {true_pt.shape}')

        bias_pt = np.mean((pred_pt-true_pt)/true_pt)
        resolution_pt = np.sqrt(np.mean(((pred_pt-true_pt)/true_pt)**2- bias_pt**2))
        LOGGER.info(f'the resolution shape is is {resolution_pt.shape}')


        # compute metrics over preprocessed amplitudes
        mse_prepd = np.mean((amp_pred.flatten() - amp_truth.flatten()) ** 2)

        results["val_loss"] = mse_prepd
        results["pt_bias"] = bias_pt
        results["pt_resolution"] = resolution_pt


        # log to mlflow
        if self.cfg.use_mlflow:
            log_dict = {
                f"eval.{title}.val.mse": mse_prepd,
            }
            for key, value in log_dict.items():
                log_mlflow(key, value)

        amp = {
            "raw": {
                "truth": amp_truth,
                "prediction": amp_pred,
                "scalar_truth": amplitudes_scalar_truth_prepd,
                "scalar_prediction": amplitudes_scalar_pred_prepd,
                "input":amp_lepton,
                "mse": mse_prepd,
                "true_pt": true_pt,
                "pred_pt": pred_pt,
            },
        }
        results["val"] = amp
        return results

    def plot(self):
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(plot_path, exist_ok=True)
        model_title = MODEL_TITLE_DICT[type(self.model.net).__name__]
        title = model_title
        LOGGER.info(f"Creating plots in {plot_path}")

        if (
            self.cfg.evaluation.save_roc
            and self.cfg.evaluate
            and ("val" in self.cfg.evaluation.eval_set)
        ):
            file = f"{plot_path}/roc.txt"
            roc = np.stack(
                (self.results["val"]["fpr"], self.results["val"]["tpr"]), axis=-1
            )
            np.savetxt(file, roc)

        plot_dict = {}
        if self.cfg.evaluate and ("val" in self.cfg.evaluation.eval_set):
            plot_dict = {"results_val": self.results["val"]}
        if self.cfg.train:
            plot_dict["train_loss"] = self.train_loss
            plot_dict["val_loss"] = self.val_loss
            plot_dict["train_lr"] = self.train_lr
            plot_dict["train_metrics"] = self.train_metrics
            plot_dict["val_metrics"] = self.val_metrics
        plot_mixer(self.cfg, plot_path, title, plot_dict)

    # overwrite _validate method to compute metrics over the full validation set
    def _validate(self, step):
        if self.ema is not None:
            with self.ema.average_parameters():
                metrics = self._evaluate_single(
                    self.val_loader, "val", step=step
                )
        else:
            metrics = self._evaluate_single(
                self.val_loader, "val", step=step
            )
        self.val_loss.append(metrics["val_loss"])
        return metrics["val_loss"]

    def _batch_loss(self, batch):
        y_pred, y_scalar, targets, targets_sc = self._get_ypred_and_label(batch)
        #print(f'the loaded label for loss is {label.shape}')
        #print(f'the pred for loss is {y_pred.shape}')
        if self.cfg.training.scalar_target:
            loss = self.loss(y_pred, targets) + self.cfg.training.lambdaloss * self.loss(y_scalar, targets_sc)
        else:
            loss = self.loss(y_pred, targets)
        assert torch.isfinite(loss).all()

        metrics = {}
        return loss, metrics

    def _get_ypred_and_label(self, batch):
        batch = batch.to(self.device)
        if self.cfg.training.scalar_target:
            targets = batch.targets.view(1,int(batch.targets.shape[0]/8), 2, 4)
            targets_sc = batch.targets_sc.view(int(batch.targets_sc.shape[0]/2), 2) 
        else:
            targets = batch.targets.view(1,int(batch.targets.shape[0]/8), 2, 4)
        embedding = embed_tagging_data_into_ga(
            batch.x, batch.scalars, batch.ptr, self.cfg.data
        )
        #print(f'the model inputs have shape {batch.x.shape}')
        y_pred, y_scalar = self.model(embedding)
        #LOGGER.info(f'the model prediction during training {y_scalar.shape}')
        #print(f'the targets during prediction are {targets.shape}')
        #y_pred = y_pred[:,0]
        return y_pred, y_scalar, targets.to(self.dtype), targets_sc.to(self.dtype) if self.cfg.training.scalar_target else None

    def _init_metrics(self):
        return {}


class TopRecoExperiment(RecoExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open_dict(self.cfg):

            # no fundamental scalar information available
            self.cfg.model.net.in_s_channels = 0

    def init_data(self):
        data_path = os.path.join(
            self.cfg.data.data_dir, f"train_TTTo2L2Nu_train_scaled.npz"
        )
        data_path_val = os.path.join(
            self.cfg.data.data_dir, f"train_TTTo2L2Nu_val_scaled.npz"
        )

        self._init_data(TopRecoDataset, data_path, data_path_val)




