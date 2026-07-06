"""Coriolis parameter and Rossby-number masking (Southern Hemisphere).

  coriolis(lat_deg) -> f
    f = 2 Ω sin(lat),  Ω = 7.2921e-5 rad/s
    Returns negative values for lat < 0 (Southern Hemisphere).
    Equatorial points (|f| < f_min) are clamped to ±f_min so safe division
    never produces inf.

  rossby_number(zeta, lat_deg) -> Ro
    Ro = |ζ| / |f|

  rossby_mask(zeta, lat_deg, threshold) -> BoolTensor
    True where Ro < threshold — marks points where QG assumptions hold.
    Used by geostrophic.py to suppress the residual in the Agulhas current
    core where Ro > 0.1.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

OMEGA = 7.2921e-5   # rad s⁻¹
F_MIN = 1e-6        # s⁻¹  — equatorial guard


def coriolis(lat_deg: Tensor) -> Tensor:
    """f = 2Ω sin(lat) [s⁻¹].  Negative in the Southern Hemisphere."""
    f = 2.0 * OMEGA * torch.sin(lat_deg * (math.pi / 180.0))
    # Guard: clamp |f| away from zero, then restore sign — single kernel vs
    # the three-tensor torch.where(f>=0, f.clamp(min=...), f.clamp(max=...)).
    return torch.copysign(f.abs().clamp(min=F_MIN), f)


def rossby_number(zeta: Tensor, lat_deg: Tensor) -> Tensor:
    """Ro = |ζ| / |f|.  Unbounded; caller should mask before using."""
    f = coriolis(lat_deg)
    return zeta.abs() / f.abs().clamp(min=F_MIN)


def rossby_mask(zeta: Tensor, lat_deg: Tensor, threshold: float = 0.1) -> Tensor:
    """Boolean mask: True where Ro < threshold (QG regime)."""
    return rossby_number(zeta, lat_deg) < threshold
