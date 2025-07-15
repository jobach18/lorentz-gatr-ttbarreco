import torch
from torch import nn


class TopRecoTransformerWrapper(nn.Module):
    """
    Standard transformer wrapper for top quark reconstruction.
    Unlike GATr, this uses standard attention mechanisms without geometric algebra.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net
        
        # Output projection to predict 2 top quarks (each with 4-momentum)
        # The transformer outputs out_channels per input jet, we need to map to 2 top quarks
        # Note: We'll use the transformer's out_channels as input to our predictor
        transformer_out_channels = getattr(net, 'linear_out', None)
        if transformer_out_channels is not None:
            out_dim = transformer_out_channels.out_features
        else:
            out_dim = net.hidden_channels  # fallback
            
        self.top_predictor = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * 4)  # 2 tops × 4 components each
        )

    def forward(self, batch):
        """
        Forward pass through the transformer
        
        Parameters:
        -----------
        batch : dict
            Contains 'x', 'mask', 'targets', 'seq_lengths'
        
        Returns:
        --------
        logits : torch.Tensor
            Predicted top quark 4-momenta, shape (batch_size, 2, 4)
        """
        x = batch['x']              # (batch_size, seq_len, 4)
        mask = batch['mask']        # (batch_size, seq_len)
        
        # For now, let's not use attention mask and handle masking in pooling
        # This avoids the dimension mismatch issue
        transformer_output = self.net(x)  # (batch_size, seq_len, out_channels)
        
        # Global pooling over sequence dimension (masked average)
        # Only average over valid (non-masked) positions
        mask_expanded = mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
        masked_output = transformer_output * mask_expanded
        pooled_output = masked_output.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)  # (batch_size, out_channels)
        
        # Predict top quarks
        top_predictions = self.top_predictor(pooled_output)  # (batch_size, 8)
        top_predictions = top_predictions.view(-1, 2, 4)     # (batch_size, 2, 4)
        
        return top_predictions


class SimpleTransformerWrapper(nn.Module):
    """
    Alternative simpler wrapper that uses the first two jets to predict the two tops
    """

    def __init__(self, net):
        super().__init__()
        self.net = net
        
        # Get output dimension from transformer
        transformer_out_channels = getattr(net, 'linear_out', None)
        if transformer_out_channels is not None:
            out_dim = transformer_out_channels.out_features
        else:
            out_dim = net.hidden_channels  # fallback
        
        # Project each jet to a top quark prediction
        self.jet_to_top = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # Each jet predicts one top's 4-momentum
        )

    def forward(self, batch):
        """
        Forward pass - use first two jets to predict two tops
        """
        x = batch['x']              # (batch_size, seq_len, 4)
        mask = batch['mask']        # (batch_size, seq_len)
        
        # Pass through transformer (without attention mask for now)
        transformer_output = self.net(x)  # (batch_size, seq_len, out_channels)
        
        # Use first two valid jets to predict the two tops
        batch_size = x.shape[0]
        top_predictions = torch.zeros(batch_size, 2, 4, device=x.device, dtype=x.dtype)
        
        for i in range(batch_size):
            valid_indices = torch.where(mask[i])[0]
            
            if len(valid_indices) >= 2:
                # Use first two jets
                top1_features = transformer_output[i, valid_indices[0]]
                top2_features = transformer_output[i, valid_indices[1]]
                
                top_predictions[i, 0] = self.jet_to_top(top1_features)
                top_predictions[i, 1] = self.jet_to_top(top2_features)
            elif len(valid_indices) == 1:
                # Only one jet available, duplicate prediction
                top_features = transformer_output[i, valid_indices[0]]
                top_pred = self.jet_to_top(top_features)
                top_predictions[i, 0] = top_pred
                top_predictions[i, 1] = top_pred
            # If no jets (shouldn't happen), leave as zeros
        
        return top_predictions
