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
        LOGGER.info(
                f"network inputs \n"
                f"multivectors={multivector.shape} and scalar={scalars.shape}"
                )

        mask = xformers_sa_mask(embedding["batch"], materialize=not self.force_xformers)
        multivector_outputs, scalar_outputs = self.net(
            multivector, scalars=scalars, attention_mask=mask
        )
        logits = self.extract_from_ga(
            multivector_outputs,
        )

        return logits

    def extract_from_ga(self, multivector):
        outputs = extract_vector(multivector[0, 1:3, :, :])
        LOGGER.info(
            f"network output with \n"
            f"multivector={multivector.shape}, outputs={outputs.shape} "
        )
        return outputs
