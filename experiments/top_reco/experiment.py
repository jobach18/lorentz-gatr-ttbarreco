import numpy as np
import torch
from torch_geometric.loader import DataLoader

import os, time
from omegaconf import open_dict

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

from experiments.base_experiment import BaseExperiment
from experiments.top_reco.dataset import TopRecoDataset
from experiments.tagging.plots import plot_mixer
from experiments.top_reco.embedding import embed_tagging_data_into_ga
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow

MODEL_TITLE_DICT = {"GATr": "GATr"}

UNITS = 40  # We use units of 40 GeV for all tagging experiments


class RecoExperiment(BaseExperiment):
    """
    Base class for Top Reco 
    """

    def _init_loss(self):
        self.loss = torch.nn.MSELoss()

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
        self.data_train.load_data(data_path)
        self.data_val.load_data(data_path_val)
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
            labels = batch.targets
            print("Data shape:", data.shape)
            print("Labels shape:", labels.shape)
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
                self.results[set_label] = self._evaluate_single(
                    loader_dict[set_label], set_label, mode="eval"
                )

    def _evaluate_single(self, loader, title, step=None):
        # compute predictions
        # note: shuffle=True or False does not matter, because we take the predictions directly from the dataloader and not from the dataset
        amplitudes_truth_prepd, amplitudes_pred_prepd = [
            [] for _ in range(self.n_datasets)
        ], [[] for _ in range(self.n_datasets)]
        LOGGER.info(f"### Starting to evaluate model on {title} dataset ###")
        self.model.eval()
        if self.cfg.training.optimizer == "ScheduleFree":
            self.optimizer.eval()
        t0 = time.time()
        for data in loader:
            for idataset, data_onedataset in enumerate(data):
                x, y = data_onedataset
                x = x.unsqueeze(0)
                pred = self.model(
                    x.to(self.device)
                )

                y_pred = pred[0, ..., 0]

                amplitudes_pred_prepd[idataset].append(y_pred.cpu().float().numpy())
                amplitudes_truth_prepd[idataset].append(
                    y.flatten().cpu().float().numpy()
                )
        amplitudes_pred_prepd = [
            np.concatenate(individual) for individual in amplitudes_pred_prepd
        ]
        amplitudes_truth_prepd = [
            np.concatenate(individual) for individual in amplitudes_truth_prepd
        ]
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
        for idataset, dataset in enumerate(self.cfg.data.dataset):
            amp_pred_prepd = amplitudes_pred_prepd[idataset]
            amp_truth_prepd = amplitudes_truth_prepd[idataset]

            # compute metrics over preprocessed amplitudes
            mse_prepd = np.mean((amp_pred_prepd - amp_truth_prepd) ** 2)

            # undo preprocessing
            amp_truth = undo_preprocess_amplitude(
                amp_truth_prepd, self.prepd_mean[idataset], self.prepd_std[idataset]
            )
            amp_pred = undo_preprocess_amplitude(
                amp_pred_prepd, self.prepd_mean[idataset], self.prepd_std[idataset]
            )

            # compute metrics over actual amplitudes
            mse = np.mean((amp_truth - amp_pred) ** 2)

            delta = (amp_truth - amp_pred) / amp_truth
            delta_maxs = [0.001, 0.01, 0.1]
            delta_rates = []
            for delta_max in delta_maxs:
                rate = np.mean(
                    (delta > -delta_max) * (delta < delta_max)
                )  # fraction of events with -delta_max < delta < delta_max
                delta_rates.append(rate)
            LOGGER.info(
                f"rate of events in delta interval on {dataset} {title} dataset:\t"
                f"{[f'{delta_rates[i]:.4f} ({delta_maxs[i]})' for i in range(len(delta_maxs))]}"
            )

            # log to mlflow
            if self.cfg.use_mlflow:
                log_dict = {
                    f"eval.{title}.{dataset}.mse": mse_prepd,
                    f"eval.{title}.{dataset}.mse_raw": mse,
                }
                for key, value in log_dict.items():
                    log_mlflow(key, value)

            amp = {
                "raw": {
                    "truth": amp_truth,
                    "prediction": amp_pred,
                    "mse": mse,
                },
                "preprocessed": {
                    "truth": amp_truth_prepd,
                    "prediction": amp_pred_prepd,
                    "mse": mse_prepd,
                },
            }
            results[dataset] = amp
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
            and ("test" in self.cfg.evaluation.eval_set)
        ):
            file = f"{plot_path}/roc.txt"
            roc = np.stack(
                (self.results["test"]["fpr"], self.results["test"]["tpr"]), axis=-1
            )
            np.savetxt(file, roc)

        plot_dict = {}
        if self.cfg.evaluate and ("test" in self.cfg.evaluation.eval_set):
            plot_dict = {"results_test": self.results["test"]}
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
                    self.val_loader, "val", mode="val", step=step
                )
        else:
            metrics = self._evaluate_single(
                self.val_loader, "val", mode="val", step=step
            )
        self.val_loss.append(metrics["loss"])
        return metrics["loss"]

    def _batch_loss(self, batch):
        y_pred, label = self._get_ypred_and_label(batch)
        loss = self.loss(y_pred, label)
        assert torch.isfinite(loss).all()

        metrics = {}
        return loss, metrics

    def _get_ypred_and_label(self, batch):
        batch = batch.to(self.device)
        embedding = embed_tagging_data_into_ga(
            batch.x, batch.scalars, batch.ptr, self.cfg.data
        )
        print(f' the embedded data is {embedding}')
        y_pred = self.model(embedding)
        print(f'unaltered ypred is {y_pred.shape}')
        y_pred = y_pred[:,0]
        return y_pred, batch.targets.to(self.dtype)

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
            self.cfg.data.data_dir, f"train_TTTo2L2Nu_train.npz"
        )
        data_path_val = os.path.join(
            self.cfg.data.data_dir, f"train_TTTo2L2Nu_val.npz"
        )

        self._init_data(TopRecoDataset, data_path, data_path_val)




