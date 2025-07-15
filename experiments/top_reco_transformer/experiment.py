import time
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import json

from experiments.base_experiment import BaseExperiment
from experiments.top_reco_transformer.dataset import TopRecoTransformerDataset, collate_fn

# Try to import logger, fallback to print if not available
try:
    from experiments.logger import LOGGER
except (ImportError, AttributeError):
    class SimpleLogger:
        def info(self, msg):
            print(f"INFO: {msg}")
        def warning(self, msg):
            print(f"WARNING: {msg}")
    LOGGER = SimpleLogger()


class TopRecoTransformerExperiment(BaseExperiment):
    """
    Top quark reconstruction experiment using standard transformer architecture.
    This is a comparison baseline to the GATr-based top_reco experiment.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_loss(self):
        """Initialize loss function"""
        self.loss = torch.nn.MSELoss()
        # Alternative: self.loss = torch.nn.SmoothL1Loss(beta=0.01)

    def init_physics(self):
        """Initialize physics-specific parameters"""
        # No special physics setup needed for standard transformer
        pass

    def init_data(self):
        """Initialize datasets"""
        train_file = os.path.join(self.cfg.data.data_dir, "train_TTTo2L2Nu_train_scaled_genbot.npz")
        val_file = os.path.join(self.cfg.data.data_dir, "train_TTTo2L2Nu_val_scaled_genbot.npz")
        
        # Use the configured files if they exist in the config
        if hasattr(self.cfg.data, 'train_file'):
            train_file = self.cfg.data.train_file
        if hasattr(self.cfg.data, 'val_file'):
            val_file = self.cfg.data.val_file
            
        self._init_data(TopRecoTransformerDataset, train_file, val_file)

    def _init_data(self, Dataset, train_path, val_path):
        """Initialize dataset objects"""
        LOGGER.info(f"Creating {Dataset.__name__} from {train_path}")
        t0 = time.time()
        
        self.data_train = Dataset()
        self.data_val = Dataset()
        
        # Load training data
        self.data_train.load_data(
            train_path, 
            data_scale=self.cfg.data.get('data_scale', None)
        )
        
        # Load validation data
        self.data_val.load_data(
            val_path, 
            data_scale=self.cfg.data.get('data_scale', None)
        )
        
        dt = time.time() - t0
        LOGGER.info(f"Finished creating datasets after {dt:.2f} s = {dt/60:.2f} min")

    def _init_dataloader(self):
        """Initialize data loaders"""
        self.train_loader = DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.training.batchsize,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        self.val_loader = DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

        LOGGER.info(
            f"Constructed dataloaders with "
            f"train_batches={len(self.train_loader)}, val_batches={len(self.val_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), {self.cfg.evaluation.batchsize} (evaluation)"
        )

    def _batch_loss(self, batch):
        """Compute loss for a single batch"""
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        pred = self.model(batch)     # (batch_size, 2, 4)
        target = batch['targets']    # (batch_size, 2, 4)
        
        # Compute loss
        loss = self.loss(pred, target)
        
        # Compute additional metrics
        with torch.no_grad():
            mse = torch.mean((pred - target) ** 2).item()
            mae = torch.mean(torch.abs(pred - target)).item()
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'loss': loss.item()
        }
        
        return loss, metrics

    def _get_ypred_and_label(self, batch):
        """Get predictions and labels for evaluation"""
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        pred = self.model(batch)
        target = batch['targets']
        return pred, target

    def evaluate(self):
        """Evaluate model on validation set"""
        self.model.eval()
        
        loader_dict = {"val": self.val_loader}
        
        # Evaluate on specified sets
        for set_label in self.cfg.evaluation.eval_set:
            if set_label not in loader_dict:
                LOGGER.warning(f"Evaluation set '{set_label}' not found, skipping")
                continue
                
            result = self._evaluate_single(loader_dict[set_label], set_label)
            
            # Save results if needed
            if hasattr(self.cfg.evaluation, 'save_results') and self.cfg.evaluation.save_results:
                self._save_results(result, set_label)
            
            # Store results
            if not hasattr(self, 'results'):
                self.results = {}
            self.results[set_label] = result
            
        return self.results.get('val', {}).get('mse', float('inf'))

    def _evaluate_single(self, loader, title):
        """Evaluate on a single dataset"""
        LOGGER.info(f"### Starting to evaluate model on {title} dataset ###")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in loader:
                pred, target = self._get_ypred_and_label(batch)
                loss = self.loss(pred, target)
                
                all_predictions.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                total_loss += loss.item()
                num_batches += 1
        
        # Concatenate all predictions and targets
        predictions = np.concatenate(all_predictions, axis=0)  # (N, 2, 4)
        targets = np.concatenate(all_targets, axis=0)          # (N, 2, 4)
        
        # Compute metrics
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        avg_loss = total_loss / num_batches
        
        # Compute per-component MSE
        mse_per_component = np.mean((predictions - targets) ** 2, axis=(0, 1))  # (4,)
        
        LOGGER.info(f"Evaluation results for {title}:")
        LOGGER.info(f"  MSE: {mse:.6f}")
        LOGGER.info(f"  MAE: {mae:.6f}")
        LOGGER.info(f"  Loss: {avg_loss:.6f}")
        LOGGER.info(f"  MSE per component (px,py,pz,E): {mse_per_component}")
        
        result = {
            'mse': mse,
            'mae': mae,
            'loss': avg_loss,
            'mse_per_component': mse_per_component.tolist(),
            'raw': {
                'predictions': predictions,
                'targets': targets
            }
        }
        
        return result

    def _save_results(self, result, set_label):
        """Save evaluation results to file matching the original GATr experiment structure"""
        # Create results directory
        os.makedirs("results_to_notebook/top_reco_transformer", exist_ok=True)
        
        # Prepare result in the same format as original experiment
        result_dict = {
            f"{set_label}_loss": f"{result['loss']:.6f}",
            "pt_bias": "0.0",  # Placeholder - could compute actual bias
            "pt_resolution": f"{result['mse']:.6f}",  # Use MSE as resolution proxy
            set_label: {
                "raw": {
                    "truth": result['raw']['targets'].tolist(),  # Ground truth targets
                    "pred": result['raw']['predictions'].tolist()  # Model predictions
                }
            }
        }
        
        # Save to JSON with same naming convention
        json_path = os.path.join("results_to_notebook/top_reco_transformer", f"result_{set_label}_lambda4.json")
        with open(json_path, "w") as json_file:
            json.dump(result_dict, json_file, default=str)
        
        LOGGER.info(f"Saved results to {json_path}")

    def _init_metrics(self):
        """Initialize training metrics"""
        return {
            'train_loss': [],
            'val_loss': [],
            'val_mse': [],
            'val_mae': [],
            'mse': [],      # Add this for training metrics
            'mae': [],      # Add this for training metrics
            'loss': []      # Add this for training metrics
        }
    
    def plot(self):
        """Generate plots for training and evaluation results"""
        from experiments.top_reco_transformer.plots import plot_transformer_results, plot_predictions_vs_truth
        
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(plot_path, exist_ok=True)
        model_title = "Transformer"
        title = model_title
        LOGGER.info(f"Creating plots in {plot_path}")

        # Prepare plot dictionary
        plot_dict = {}
        if self.cfg.evaluate and ("val" in self.cfg.evaluation.eval_set):
            plot_dict["results_val"] = self.results["val"]
        if self.cfg.train:
            plot_dict["train_loss"] = self.train_loss
            plot_dict["val_loss"] = self.val_loss
            plot_dict["train_lr"] = self.train_lr
            plot_dict["train_metrics"] = self.train_metrics
            plot_dict["val_metrics"] = self.val_metrics

        # Generate plots
        plot_transformer_results(self.cfg, plot_path, title, plot_dict)
        
        # Plot predictions vs truth if evaluation data is available
        if (self.cfg.evaluate and 
            ("val" in self.cfg.evaluation.eval_set) and 
            hasattr(self, 'results') and 
            'val' in self.results and 
            'raw' in self.results['val']):
            
            predictions = self.results['val']['raw']['predictions']
            targets = self.results['val']['raw']['targets']
            
            # Convert to numpy if needed
            if hasattr(predictions, 'numpy'):
                predictions = predictions
            if hasattr(targets, 'numpy'):
                targets = targets
                
            plot_predictions_vs_truth(
                self.cfg, plot_path, predictions, targets, title
            )
        
        LOGGER.info(f"Finished creating plots in {plot_path}")
