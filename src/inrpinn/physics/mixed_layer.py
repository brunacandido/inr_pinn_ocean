"""EQ-11 — Mixed layer heat budget (surface boundary condition).

Residual at z ≤ MLD:
  ρ cp h ∂T_ml/∂t = Q_net − Q_pen(h)

Q_net is unavailable from GLORYS12V1, so the residual is implemented as a
temporal smoothness constraint in the near-surface layer: the mixed-layer
temperature should vary smoothly in time with a magnitude bounded by the
seasonal scale (~1 °C/month).  Specifically:

  L_ml = mean_surface( ReLU(|∂CT/∂t| − ∂CT_max) )

where ∂CT_max is the maximum physically plausible SST tendency.
Applied to depths < mixed_layer_depth_m (default 200 m as a proxy for MLD).
"""
from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from .eos import _phys_coords

# Maximum physically plausible SST tendency [°C/day]
# ~3 °C/month ≈ 0.1 °C/day in the Agulhas region
_DCT_DT_MAX = 0.10


def mixed_layer_residual(
    grads: dict[str, Tensor],
    coords: Tensor,
    bounds: dict,
    mixed_layer_depth_m: float = 200.0,
) -> Tensor:
    """EQ-11: near-surface temperature tendency soft bound.

    Applies ReLU penalty where |∂CT/∂t| exceeds the expected seasonal rate.
    Only activates for points shallower than mixed_layer_depth_m.

    Parameters
    ----------
    mixed_layer_depth_m : depth threshold [m] for the mixed-layer mask.
    """
    _, _, depth_m, _ = _phys_coords(coords, bounds)

    dCT  = bounds['T'][1]    - bounds['T'][0]       # physical CT range
    dt   = bounds['time'][1] - bounds['time'][0]    # time range [days]

    # ∂CT/∂t in physical units [°C/day]
    dCT_dt_phys = grads['dCT_dt'] * (dCT / max(dt, 1.0))

    mask = depth_m < mixed_layer_depth_m
    # Stay on device: masked weighted mean avoids the GPU→CPU sync of mask.any()
    # and the scatter-gather of boolean fancy indexing.
    m = mask.float()
    excess = F.relu(dCT_dt_phys.abs() - _DCT_DT_MAX)
    return (excess * m).sum() / m.sum().clamp(min=1.0)
