import torch
import numpy as np
import os

# Try to import logger, fallback to print if not available
try:
    from experiments.logger import LOGGER
except (ImportError, AttributeError):
    class SimpleLogger:
        def info(self, msg):
            print(f"INFO: {msg}")
    LOGGER = SimpleLogger()


class TopRecoTransformerDataset(torch.utils.data.Dataset):
    """
    Dataset for top reconstruction using standard transformer architecture.
    Unlike the original top_reco which uses torch_geometric batching,
    this uses standard PyTorch batching with padding for variable-length sequences.
    
    Structure of the elements in self.data_list:
    x : torch.tensor of shape (num_jets, 4)
        List of 4-momenta of jet constituents/jets
    targets : torch.tensor of shape (8,) -> reshaped to (2, 4)
        Target 4-vectors for reconstructed top quarks
    """

    def __init__(self):
        super().__init__()
        self.data_list = []

    def load_data(
        self,
        filename,
        data_scale=None,
        dtype=torch.float32,
        max_jets=None,
    ):
        """
        Load data from npz file and convert to standard tensor format
        
        Parameters:
        -----------
        filename : str
            Path to the data file
        data_scale : str or None
            Scaling method to apply ('minkNorm' or None)
        dtype : torch.dtype
            Data type for tensors
        max_jets : int or None
            Maximum number of jets per event (for padding)
        """
        LOGGER.info(f"Loading data from {filename}")
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Data file not found: {filename}")
        
        data = np.load(filename)
        
        # Load kinematics and targets
        kinematics = data["x"]  # Shape: (N, num_jets, 4)
        targets = data["y"]     # Shape: (N, 8) - flattened (top1_px, top1_py, top1_pz, top1_E, top2_px, ...)
        
        # Convert to tensors
        kinematics = torch.tensor(kinematics, dtype=dtype)
        targets = torch.tensor(targets, dtype=dtype)
        
        # Reshape targets to (N, 2, 4) - 2 top quarks, each with 4-momentum
        targets = targets.view(-1, 2, 4)
        
        # Apply Minkowski normalization if specified
        if data_scale == 'minkNorm':
            kinematics, targets = self._apply_minkowski_scaling(kinematics, targets)
        
        # Determine max_jets if not provided
        if max_jets is None:
            max_jets = max(self._count_nonzero_jets(kinematics[i]) for i in range(len(kinematics)))
            LOGGER.info(f"Determined max_jets = {max_jets}")
        
        # Process each event
        for i in range(kinematics.shape[0]):
            # Remove zero-padded jets (assuming padding is all zeros)
            valid_mask = self._get_valid_jets_mask(kinematics[i])
            valid_jets = kinematics[i][valid_mask]
            
            self.data_list.append({
                'x': valid_jets,          # Shape: (num_valid_jets, 4)
                'targets': targets[i],    # Shape: (2, 4)
                'num_jets': len(valid_jets)
            })
        
        LOGGER.info(f"Loaded {len(self.data_list)} events with targets of shape {targets.shape}")

    def _apply_minkowski_scaling(self, kinematics, targets):
        """Apply Minkowski normalization scaling matching the original top_reco experiment"""
        EPS = 1e-5
        
        def minkowski_norm(vectors):
            # Following the original implementation exactly
            # Format: (E, px, py, pz) where E is first component
            E = vectors[..., 0]  # Energy component
            p = vectors[..., 1:]  # Momentum components (px, py, pz)
            norm = torch.sqrt(torch.clamp(E**2 - p.pow(2).sum(dim=-1), min=EPS))
            return norm.unsqueeze(-1)
        
        # Scale kinematics: shape (N, num_jets, 4)
        kin_norms = 0.25 * minkowski_norm(kinematics)
        kinematics = kinematics / kin_norms
        
        # Scale targets: shape (N, 2, 4)  
        tar_norms = 0.25 * minkowski_norm(targets)
        targets = targets / tar_norms
        
        return kinematics, targets

    def _get_valid_jets_mask(self, jets):
        """Get mask for non-zero jets"""
        # Consider a jet valid if it has non-zero energy or any non-zero momentum component
        return (jets.abs().sum(-1) > 1e-6)

    def _count_nonzero_jets(self, jets):
        """Count number of non-zero jets in an event"""
        return self._get_valid_jets_mask(jets).sum().item()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    Pads sequences to the maximum length in the batch.
    
    Parameters:
    -----------
    batch : list of dict
        List of data samples
    
    Returns:
    --------
    dict with keys:
        'x': padded input sequences (batch_size, max_seq_len, 4)
        'targets': target tensors (batch_size, 2, 4)
        'mask': attention mask (batch_size, max_seq_len)
        'seq_lengths': actual sequence lengths (batch_size,)
    """
    batch_size = len(batch)
    max_seq_len = max(item['num_jets'] for item in batch)
    
    # Initialize padded tensors
    x_padded = torch.zeros(batch_size, max_seq_len, 4, dtype=batch[0]['x'].dtype)
    targets = torch.stack([item['targets'] for item in batch])
    mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    seq_lengths = torch.tensor([item['num_jets'] for item in batch], dtype=torch.long)
    
    # Fill in the actual data
    for i, item in enumerate(batch):
        seq_len = item['num_jets']
        x_padded[i, :seq_len] = item['x']
        mask[i, :seq_len] = True
    
    return {
        'x': x_padded,
        'targets': targets,
        'mask': mask,
        'seq_lengths': seq_lengths
    }
