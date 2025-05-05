import torch
from torch import nn
from torch_geometric.nn.aggr import MeanAggregation

from gatr.interface import extract_scalar
from gatr.interface import extract_vector
from xformers.ops.fmha import BlockDiagonalMask
from experiments.logger import LOGGER


def xformers_sa_mask(batch, materialize=False):
    """
    Construct attention mask that makes sure that objects only attend to each other
    within the same batch element, and not across batch elements

    Parameters
    ----------
    batch: torch.tensor
        batch object in the torch_geometric.data naming convention
        contains batch index for each event in a sparse tensor
    materialize: bool
        Decides whether a xformers or ('materialized') torch.tensor mask should be returned
        The xformers mask allows to use the optimized xformers attention kernel, but only runs on gpu

    Returns
    -------
    mask: xformers.ops.fmha.attn_bias.BlockDiagonalMask or torch.tensor
        attention mask, to be used in xformers.ops.memory_efficient_attention
        or torch.nn.functional.scaled_dot_product_attention
    """
    bincounts = torch.bincount(batch).tolist()
    mask = BlockDiagonalMask.from_seqlens(bincounts)
    if materialize:
        # materialize mask to torch.tensor (only for testing purposes)
        mask = mask.materialize(shape=(len(batch), len(batch))).to(batch.device)
    return mask


class RecoGATrWrapper(nn.Module):
    """
    L-GATr for top quark system reco
    """

    def __init__(
        self,
        net,
        force_xformers=True,
    ):
        super().__init__()
        self.net = net
        self.force_xformers = force_xformers

    def forward(self, embedding):
        multivector = embedding["mv"].unsqueeze(0)
        scalars = embedding["s"].unsqueeze(0)

        mask = xformers_sa_mask(embedding["batch"], materialize=not self.force_xformers)
        multivector_outputs, scalar_outputs = self.net(
            multivector, scalars=scalars, attention_mask=mask
        )
        logits = self.extract_from_ga(
            multivector_outputs
        )

        return logits

    def extract_from_ga(self, multivector):
        #summed_mv = sum_along_dimension(multivector)
        print(f'the multivector is {multivector.shape}')
        outputs = extract_vector(multivector[0,::14,:,:])
        print(f'the output shape is: {outputs.shape}')
        return outputs

def sum_along_dimension(tensor):
    # Dimension along which to sum every 7 elements
    dim = 1

    # Ensure the dimension size is divisible by 7
    size_to_trim = (tensor.size(dim) // 7) * 7
    trimmed_tensor = tensor.narrow(dim, 0, size_to_trim)

    # Reshape and sum along the specified dimension
    new_shape = list(trimmed_tensor.shape)
    new_shape[dim] = -1  # Replace the size of the target dimension with -1 (batch groups)
    new_shape.insert(dim + 1, 7)  # Insert a new dimension of size 7
    reshaped_tensor = trimmed_tensor.view(*new_shape)
    summed_tensor = reshaped_tensor.sum(dim=dim + 1)
    return summed_tensor
