"""EQ-10 — Temperature advection–diffusion (vertical term).

Residual:
  ∂T/∂t + u·∇T = κ_T ∂²T/∂z²

Horizontal diffusion omitted at GLORYS 1/12° resolution.

∂²T/∂z² is computed via second-order autograd on grads['dCT_dz'].
Requires coords to have requires_grad=True when passed to coords_and_grad.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .eos import _phys_coords
from .thermal_wind import _density_gradients
from ..utils.coriolis import coriolis

G    = 9.81
RHO0 = 1025.0


def temp_adv_residual(
    grads: dict[str, Tensor],
    coords: Tensor,
    bounds: dict,
    kappa_T: float = 1e-5,
    depth_min: float = 0.0,
) -> Tensor:
    """EQ-10: temperature advection–diffusion  ∂CT/∂t + u·∇CT = κ_T ∂²CT/∂z².

    Parameters
    ----------
    kappa_T   : vertical thermal diffusivity [m²/s].
    depth_min : exclude shallow depths [m] above the mixed layer.

    Notes
    -----
    coords must have requires_grad=True (set by PINN before coords_and_grad).
    """
    _, lat, depth_m, _ = _phys_coords(coords, bounds)

    dCT  = bounds['T'][1]     - bounds['T'][0]
    ddep = bounds['depth'][1] - bounds['depth'][0]
    dlat = bounds['lat'][1]   - bounds['lat'][0]
    dlon = bounds['lon'][1]   - bounds['lon'][0]

    lat_rad      = lat * (math.pi / 180.0)
    deg_to_m_lat = math.pi / 180.0 * 6_371_000.0
    deg_to_m_lon = deg_to_m_lat * lat_rad.cos().clamp(min=1e-6)

    dCT_dt = grads['dCT_dt'] * (dCT / (bounds['time'][1] - bounds['time'][0]))
    dCT_dx = grads['dCT_dx'] * (dCT / dlon) / deg_to_m_lon
    dCT_dy = grads['dCT_dy'] * (dCT / dlat) / deg_to_m_lat

    d2CT_dz2_norm = torch.autograd.grad(
        grads['dCT_dz'].sum(), coords, create_graph=True, retain_graph=True
    )[0][:, 2]
    d2CT_dz2_phys = 2.0 * dCT / ddep ** 2 * d2CT_dz2_norm

    # Inline geostrophic velocity — avoids a second _phys_coords + _density_gradients
    # call that would otherwise happen inside geostrophic_velocity().
    drho_dx, drho_dy, _ = _density_gradients(grads, coords, bounds)
    f   = coriolis(lat)
    u_g =  (G / (f * RHO0)) * drho_dy * depth_m
    v_g = -(G / (f * RHO0)) * drho_dx * depth_m

    mask  = depth_m > depth_min
    m     = mask.float()
    scale = dCT / max(bounds['time'][1] - bounds['time'][0], 1.0)
    residual = dCT_dt + u_g * dCT_dx + v_g * dCT_dy - kappa_T * d2CT_dz2_phys
    return ((residual / scale).pow(2) * m).sum() / m.sum().clamp(min=1.0)
