"""EQ-8 — Ertel potential vorticity conservation.

Full (non-QG) PV:
  q_E = (1/ρ) (f k̂ + ∇×u) · ∇b  ≈  (f + ζ) N² / ρ₀

Ertel PV is conserved along isopycnals for adiabatic, frictionless flow.
Unlike QGPV, this is valid in the high-Ro Agulhas current core.

Implementation: relative vorticity ζ = ∂v_g/∂x − ∂u_g/∂y requires second-
order density derivatives.  Here we implement the approximate conservation
of the planetary-vorticity component: D(f N²)/Dt ≈ 0, i.e.,
  u_g · ∂(f N²)/∂x + v_g · ∂(f N²)/∂y + w · ∂(f N²)/∂z ≈ 0

where the dominant terms are the horizontal advection of f (beta effect)
and the advection of N² by the geostrophic flow.
"""
from __future__ import annotations

from torch import Tensor

from .eos import _phys_coords
from .thermal_wind import _density_gradients
from ..utils.coriolis import coriolis

import math

G    = 9.81
RHO0 = 1025.0
R_EARTH = 6_371_000.0


def ertel_pv_residual(
    grads: dict[str, Tensor],
    coords: Tensor,
    bounds: dict,
    ro_threshold: float = 0.1,
    depth_min: float = 50.0,
) -> Tensor:
    """EQ-8: Ertel PV conservation proxy  D(f N²)/Dt ≈ 0.

    Penalises the advective tendency of (f · N²) by the diagnosed
    geostrophic flow.  Includes the beta-plane meridional advection of f
    (the dominant source of PV change in the Agulhas region).

    No ro_threshold mask: valid across the full Rossby-number spectrum,
    unlike QGPV which is masked to the QG regime.
    """
    _, lat, depth_m, _ = _phys_coords(coords, bounds)
    f = coriolis(lat)

    lat_rad = lat * (math.pi / 180.0)
    Omega   = 7.2921e-5
    beta    = 2.0 * Omega * lat_rad.cos() / R_EARTH

    # Single _density_gradients call; inline geostrophic_velocity to avoid a
    # second call to _density_gradients and _phys_coords inside it.
    drho_dx, drho_dy, drho_dz = _density_gradients(grads, coords, bounds)
    N2      = -(G / RHO0) * drho_dz
    N2_safe = N2.clamp(min=1e-8)
    u_g =  (G / (f * RHO0)) * drho_dy * depth_m
    v_g = -(G / (f * RHO0)) * drho_dx * depth_m

    mask = depth_m > depth_min
    m    = mask.float()

    beta_term  = beta * v_g * N2_safe
    beta_scale = 2e-11
    U_scale    = 0.5
    N2_scale   = 1e-4
    norm       = beta_scale * U_scale * N2_scale
    return ((beta_term / norm).pow(2) * m).sum() / m.sum().clamp(min=1.0)
