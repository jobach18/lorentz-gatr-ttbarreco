# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
from functools import lru_cache

import torch
import math

from gatr.primitives.linear import grade_project
from gatr.utils.einsum import cached_einsum


@lru_cache()
def _load_inner_product_factors(device=torch.device("cpu")) -> torch.Tensor:
    """Constructs an array of 1's and -1's for the metric of the space,
    used to compute the inner product.

    Parameters
    ----------
    device : torch.device
        Device

    Returns
    -------
    ip_factors : torch.Tensor with shape (16,)
        Inner product factors
    """
    
    _INNER_PRODUCT_FACTORS = [1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1]
    factors = torch.tensor(_INNER_PRODUCT_FACTORS, dtype=torch.float32, device=torch.device("cpu")).to_dense()
    return factors

@lru_cache()
def _load_metric_grades(device=torch.device("cpu")) -> torch.Tensor:
    """Generate tensor of the diagonal of the GA metric, combined with a grade projection.

    Parameters
    ----------
    device : torch.device
        Device

    Returns
    -------
    torch.Tensor of shape [5, 16]
    """
    m = _load_inner_product_factors(device)
    m_grades = torch.zeros(5, 16, device=device)
    offset = 0
    for k in range(4 + 1):
        d = math.comb(4, k)
        m_grades[k, offset : offset + d] = m[offset : offset + d]
        offset += d
    return m_grades

def inner_product(x: torch.Tensor, y: torch.Tensor, channel_sum: bool = False) -> torch.Tensor:
    """Computes the inner product of multivectors f(x,y) = <x, y> = <~x y>_0.

    In addition to summing over the 16 multivector dimensions, this function also sums
    over an additional channel dimension if channel_sum == True.

    Equal to `geometric_product(reverse(x), y)[..., [0]]` (but faster).

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16) or (..., channels, 16)
        First input multivector. Batch dimensions must be broadcastable between x and y.
    y : torch.Tensor with shape (..., 16) or (..., channels, 16)
        Second input multivector. Batch dimensions must be broadcastable between x and y.
    channel_sum: bool
        Whether to sum over the second-to-last axis (channels)

    Returns
    -------
    outputs : torch.Tensor with shape (..., 1)
        Result. Batch dimensions are result of broadcasting between x and y.
    """

    x = x * _load_inner_product_factors()

    if channel_sum:
        outputs = cached_einsum("... c i, ... c i -> ...", x, y)
    else:
        outputs = cached_einsum("... i, ... i -> ...", x, y)

    # We want the output to have shape (..., 1)
    outputs = outputs.unsqueeze(-1)

    return outputs


def squared_norm(x: torch.Tensor) -> torch.Tensor:
    """Computes the squared GA norm of an input multivector.

    Equal to inner_product(x, x).

    NOTE: this primitive is not used widely in our architectures.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 1)
        Geometric algebra norm of x.
    """

    return inner_product(x, x)


def pin_invariants(x: torch.Tensor) -> torch.Tensor:
    """Computes five invariants from multivectors: scalar component, norms of the four other grades.

    NOTE: this primitive is not used widely in our architectures.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 5)
        Invariants computed from input multivectors
    """

    # Project to grades
    projections = grade_project(x)  # (..., 5, 16)

    # Compute norms
    squared_norms = inner_product(projections, projections)[..., 0]  # (..., 5)
    norms = torch.sqrt(torch.clamp(squared_norms, 0.0))

    # Outputs: scalar component of input and norms of four other grades
    return torch.cat((x[..., [0]], norms[..., 1:]), dim=-1)  # (..., 5)

def abs_squared_norm(x: torch.Tensor) -> torch.Tensor:
    """Computes a modified version of the squared norm that is positive semidefinite and can
    therefore be used in layer normalization. 

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 1)
        Geometric algebra norm of x.
    """
    m = _load_metric_grades(device=x.device)
    abs_squared_norms = cached_einsum("... i, ... i, g i -> ... g", x, x, m).abs().sum(-1, keepdim=True)
    return abs_squared_norms
