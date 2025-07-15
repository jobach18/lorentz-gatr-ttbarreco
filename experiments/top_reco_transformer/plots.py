import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages

# turn off warnings from 1/0 when plotting roc curves
import warnings
warnings.simplefilter("ignore", RuntimeWarning)

from experiments.base_plots import plot_loss, plot_metric

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams[
    "text.latex.preamble"
] = r"\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath} \usepackage{siunitx}"

FONTSIZE = 14
FONTSIZE_LEGEND = 13
FONTSIZE_TICK = 12

colors = ["black", "#0343DE", "#A52A2A", "darkorange"]


def plot_transformer_results(cfg, plot_path, title, plot_dict):
    """
    Plot results for transformer top reconstruction experiment
    
    Parameters:
    -----------
    cfg : config
        Experiment configuration
    plot_path : str
        Directory to save plots
    title : str
        Title for plots
    plot_dict : dict
        Dictionary containing training and evaluation results
    """
    
    # Plot training loss
    if cfg.plotting.loss and cfg.train:
        file = f"{plot_path}/loss.pdf"
        with PdfPages(file) as out:
            plot_loss(
                out,
                [plot_dict["train_loss"], plot_dict["val_loss"]],
                plot_dict["train_lr"],
                labels=["train loss", "val loss"],
                logy=True,
            )
    
    # Plot training metrics (MSE, MAE)
    if cfg.train and "train_metrics" in plot_dict:
        # MSE plot
        if "mse" in plot_dict["train_metrics"] and len(plot_dict["train_metrics"]["mse"]) > 0:
            file = f"{plot_path}/mse.pdf"
            with PdfPages(file) as out:
                train_mse = plot_dict["train_metrics"]["mse"]
                val_mse = plot_dict.get("val_metrics", {}).get("mse", [])
                if len(val_mse) > 0:
                    plot_metric(
                        out,
                        [train_mse, val_mse],
                        "MSE",
                        labels=["train MSE", "val MSE"],
                        logy=True
                    )
                else:
                    plot_metric(
                        out,
                        [train_mse],
                        "MSE",
                        labels=["train MSE"],
                        logy=True
                    )
        
        # MAE plot
        if "mae" in plot_dict["train_metrics"] and len(plot_dict["train_metrics"]["mae"]) > 0:
            file = f"{plot_path}/mae.pdf"
            with PdfPages(file) as out:
                train_mae = plot_dict["train_metrics"]["mae"]
                val_mae = plot_dict.get("val_metrics", {}).get("mae", [])
                if len(val_mae) > 0:
                    plot_metric(
                        out,
                        [train_mae, val_mae],
                        "MAE",
                        labels=["train MAE", "val MAE"],
                        logy=False
                    )
                else:
                    plot_metric(
                        out,
                        [train_mae],
                        "MAE",
                        labels=["train MAE"],
                        logy=False
                    )


def plot_predictions_vs_truth(cfg, plot_path, predictions, targets, title=""):
    """
    Plot predictions vs truth for top quark reconstruction
    
    Parameters:
    -----------
    predictions : np.array
        Shape (N, 2, 4) - predicted top 4-momenta
    targets : np.array  
        Shape (N, 2, 4) - true top 4-momenta
    """
    component_labels = ["px", "py", "pz", "E"]
    
    file = f"{plot_path}/predictions_vs_truth.pdf"
    with PdfPages(file) as out:
        # Create 2x4 subplot for 2 tops × 4 components
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"Predictions vs Truth - {title}", fontsize=FONTSIZE)
        
        for top_idx in range(2):
            for comp_idx in range(4):
                ax = axes[top_idx, comp_idx]
                
                pred_values = predictions[:, top_idx, comp_idx].flatten()
                true_values = targets[:, top_idx, comp_idx].flatten()
                
                # Sample a subset for plotting if too many points
                if len(pred_values) > 10000:
                    indices = np.random.choice(len(pred_values), 10000, replace=False)
                    pred_values = pred_values[indices]
                    true_values = true_values[indices]
                
                ax.scatter(true_values, pred_values, alpha=0.5, s=1)
                
                # Add diagonal line for perfect prediction
                min_val = min(true_values.min(), pred_values.min())
                max_val = max(true_values.max(), pred_values.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
                
                ax.set_xlabel(f"True {component_labels[comp_idx]}", fontsize=FONTSIZE_TICK)
                ax.set_ylabel(f"Pred {component_labels[comp_idx]}", fontsize=FONTSIZE_TICK)
                ax.set_title(f"Top {top_idx+1} - {component_labels[comp_idx]}", fontsize=FONTSIZE_TICK)
                
                # Add correlation coefficient
                corr = np.corrcoef(true_values, pred_values)[0, 1]
                ax.text(0.05, 0.95, f"r={corr:.3f}", transform=ax.transAxes, 
                       fontsize=FONTSIZE_TICK, verticalalignment='top')
        
        plt.tight_layout()
        out.savefig(fig, bbox_inches="tight")
        plt.close(fig)
