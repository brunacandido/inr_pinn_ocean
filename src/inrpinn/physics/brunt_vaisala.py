"""EQ-7 — Brunt–Väisälä frequency / static stability regularisation.

N² = -(g/ρ₀) ∂ρ/∂z

This must be positive everywhere for a statically stable water column.
Enforced as a soft inequality constraint:
  L_N2 = mean( ReLU(-N²) )   (penalises negative values only)

A physically unstable column (N² < 0) would indicate convective instability,
which cannot be represented by the smooth SIREN field.  This regularisation
prevents the network from producing unrealistic inversions of the density
profile, especially important for the AAIW salinity minimum at 600–1200 m.
"""
from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from .thermal_wind import _density_gradients

G    = 9.81
RHO0 = 1025.0


def brunt_vaisala_residual(
    grads: dict[str, Tensor],
    coords: Tensor,
    bounds: dict,
) -> Tensor:
    """EQ-7: static stability soft constraint  mean(ReLU(-N²)).

    Returns 0 when the whole collocation batch is stably stratified (N² ≥ 0).
    Penalises each unstable point proportionally to its |N²| deficit.
    """
    _, _, drho_dz = _density_gradients(grads, coords, bounds)
    N2 = -(G / RHO0) * drho_dz   # s⁻²  (positive = stable)
    return F.relu(-N2).mean()
