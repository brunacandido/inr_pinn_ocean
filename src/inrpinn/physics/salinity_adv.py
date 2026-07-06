"""EQ-9 — Salinity conservation (advection–diffusion).

Residual:
  ∂S/∂t + u·∇S = κ_S ∂²S/∂z²

u = (u_g, v_g) diagnosed geostrophic velocity.  Horizontal diffusion is
omitted at GLORYS 1/12° resolution.

∂²S/∂z² is computed via second-order autograd on grads['dSA_dz'].
Requires coords to have requires_grad=True when passed to coords_and_grad.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .eos import _phys_coords
from .thermal_wind import _density_gradients
from ..utils.coriolis import coriolis

G       = 9.81
RHO0    = 1025.0


def salinity_adv_residual(
    grads: dict[str, Tensor],
    coords: Tensor,
    bounds: dict,
    kappa_S: float = 1e-5,
    depth_min: float = 0.0,
) -> Tensor:
    """EQ-9: salinity advection–diffusion  ∂SA/∂t + u·∇SA = κ_S ∂²SA/∂z².

    Parameters
    ----------
    kappa_S   : vertical salinity diffusivity [m²/s].
    depth_min : exclude shallow depths [m] where surface forcing dominates.

    Notes
    -----
    coords must have requires_grad=True (set by PINN before coords_and_grad).
    """
    _, lat, depth_m, _ = _phys_coords(coords, bounds)

    dSA  = bounds['S'][1]     - bounds['S'][0]
    ddep = bounds['depth'][1] - bounds['depth'][0]
    dlat = bounds['lat'][1]   - bounds['lat'][0]
    dlon = bounds['lon'][1]   - bounds['lon'][0]

    lat_rad      = lat * (math.pi / 180.0)
    deg_to_m_lat = math.pi / 180.0 * 6_371_000.0
    deg_to_m_lon = deg_to_m_lat * lat_rad.cos().clamp(min=1e-6)

    dSA_dt = grads['dSA_dt'] * (dSA / (bounds['time'][1] - bounds['time'][0]))
    dSA_dx = grads['dSA_dx'] * (dSA / dlon) / deg_to_m_lon
    dSA_dy = grads['dSA_dy'] * (dSA / dlat) / deg_to_m_lat

    d2SA_dz2_norm = torch.autograd.grad(
        grads['dSA_dz'].sum(), coords, create_graph=True, retain_graph=True
    )[0][:, 2]
    d2SA_dz2_phys = 2.0 * dSA / ddep ** 2 * d2SA_dz2_norm

    # Inline geostrophic velocity — avoids a second _phys_coords + _density_gradients
    # call that would otherwise happen inside geostrophic_velocity().
    drho_dx, drho_dy, _ = _density_gradients(grads, coords, bounds)
    f   = coriolis(lat)
    u_g =  (G / (f * RHO0)) * drho_dy * depth_m
    v_g = -(G / (f * RHO0)) * drho_dx * depth_m

    mask  = depth_m > depth_min
    m     = mask.float()
    scale = dSA / max(bounds['time'][1] - bounds['time'][0], 1.0)
    residual = dSA_dt + u_g * dSA_dx + v_g * dSA_dy - kappa_S * d2SA_dz2_phys
    return ((residual / scale).pow(2) * m).sum() / m.sum().clamp(min=1.0)
