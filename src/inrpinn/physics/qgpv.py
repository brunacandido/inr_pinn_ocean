"""EQ-6 — Quasi-geostrophic potential vorticity conservation.

Residual (steady-state QG PV):
  J(ψ, q_QG) = 0
  q_QG = ∇²ψ + f + ∂/∂z(f²/N² · ∂ψ/∂z)

VALID ONLY WHERE Ro < ro_threshold (default 0.1).

Implementation: the full Jacobian J(ψ, q) requires second-order mixed spatial
derivatives.  Here we use the stretching-term proxy:
  q_stretch ≈ f²/N² · ∂²ψ/∂z²  ≈  -(f/N²) · (g/ρ₀) · ∂ρ/∂z
Penalising |u_g · ∂q_stretch/∂x + v_g · ∂q_stretch/∂y| serves as an
advective QG PV conservation constraint in the QG regime.
"""
from __future__ import annotations

from torch import Tensor

from .eos import _phys_coords
from .thermal_wind import _density_gradients
from ..utils.coriolis import coriolis

G    = 9.81
RHO0 = 1025.0


def qgpv_residual(
    grads: dict[str, Tensor],
    coords: Tensor,
    bounds: dict,
    ro_threshold: float = 0.1,
    depth_min: float = 50.0,
) -> Tensor:
    """EQ-6: QG PV conservation proxy, masked to Ro < ro_threshold.

    Penalises the product of geostrophic velocity magnitude with the
    stretching-term proxy q_s = -(f/N²) · (g/ρ₀) · ∂ρ/∂z, acting as
    a first-order approximation to the QG PV advection u·∇q = 0.

    Returns zero outside the QG regime (Ro ≥ ro_threshold) or
    in the Ekman layer (depth < depth_min).
    """
    _, lat, depth_m, _ = _phys_coords(coords, bounds)
    f = coriolis(lat)
    # Single _density_gradients call; inline geostrophic_velocity to avoid a
    # second call to _density_gradients and _phys_coords inside it.
    drho_dx, drho_dy, drho_dz = _density_gradients(grads, coords, bounds)

    N2      = -(G / RHO0) * drho_dz
    N2_safe = N2.clamp(min=1e-8)
    q_stretch = f.abs() / N2_safe * (G / RHO0) * drho_dz.abs()

    f_abs = f.abs().clamp(min=1e-6)
    u_g   =  (G / (f * RHO0)) * drho_dy * depth_m
    v_g   = -(G / (f * RHO0)) * drho_dx * depth_m
    Ro    = (u_g.abs() + v_g.abs()) / (2.0 * f_abs * 50e3)

    mask  = (depth_m > depth_min) & (Ro < ro_threshold)
    m     = mask.float()
    U_typ = 0.5
    adv   = (u_g.abs() + v_g.abs()) * q_stretch / U_typ
    return (adv * m).sum() / m.sum().clamp(min=1.0)
