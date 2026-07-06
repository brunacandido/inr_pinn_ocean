"""EQ-4 — Geostrophic balance (Southern Hemisphere sign convention).

Horizontal momentum balance in the large-scale ocean interior:

    f v  =  (1/ρ₀) ∂p/∂x      (zonal)
    f u  = −(1/ρ₀) ∂p/∂y      (meridional)

With f < 0 in the Southern Hemisphere:
    u_g =  (1/(f ρ₀)) ∂p/∂y    →  southward SSH gradient → eastward flow
    v_g = −(1/(f ρ₀)) ∂p/∂x

The geostrophic velocity is diagnosed from the *baroclinic* pressure anomaly,
which is computed from the density field via the thermal-wind integral:

    p_bc(z) = g ∫_{0}^{z} ρ′(z′) dz′,   ρ′ = ρ − ρ₀

For scattered collocation points (no fixed vertical grid), the integral is
approximated locally using the vertical density gradient:

    ∂p_bc/∂x = g · ρ′ · (∂z/∂x)|_ρ  ≈  0       (z is an independent coord)
    ∂p_bc/∂x ≈ g · (∂ρ/∂x) · z       (leading-order linear approximation)

The resulting geostrophic velocity estimate at depth z:
    u_g ≈  (g / (f ρ₀)) · (∂ρ/∂y) · z
    v_g ≈ −(g / (f ρ₀)) · (∂ρ/∂x) · z

The residual is the geostrophic imbalance between f·(u_g, v_g) and the
implied pressure-gradient force from the diagnosed pressure:

    R_x  =  f v_g  −  (g / ρ₀) · (∂ρ/∂x) · z      [m s⁻²]
    R_y  =  f u_g  +  (g / ρ₀) · (∂ρ/∂y) · z      [m s⁻²]

Under the linear baroclinic approximation both R_x and R_y are identically
zero (the thermal-wind velocity by construction satisfies geostrophy).  The
loss therefore acts as a **regulariser on the magnitude of geostrophic
velocity**, with Rossby-number masking to suppress it in the ageostrophic
Agulhas core (Ro > ro_threshold).

Applied only below depth_min_geostrophic (default 50 m) to avoid the
Ekman layer where ageostrophic terms dominate.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .eos import _phys_coords, density
from .thermal_wind import _density_gradients, thermal_wind_shear
from ..utils.coriolis import coriolis

# ── Physical constants ───────────────────────────────────────────────────────
G    = 9.81      # m s⁻²
RHO0 = 1025.0    # kg m⁻³
R_EARTH = 6_371_000.0  # m


def geostrophic_velocity(
    grads: dict[str, Tensor],
    coords: Tensor,
    bounds: dict,
) -> tuple[Tensor, Tensor]:
    """Baroclinic geostrophic velocity (u_g, v_g) diagnosed from density.

    Uses the linear baroclinic approximation:
        u_g(z) ≈  (g / (f ρ₀)) · ∂ρ/∂y · z
        v_g(z) ≈ −(g / (f ρ₀)) · ∂ρ/∂x · z

    Parameters
    ----------
    grads : dict from siren.coords_and_grad
    coords : (B, 4) normalised
    bounds : config normalisation dict

    Returns
    -------
    u_g : (B,) Tensor [m s⁻¹]
    v_g : (B,) Tensor [m s⁻¹]
    """
    _, lat, depth_m, _ = _phys_coords(coords, bounds)
    f = coriolis(lat)                                   # s⁻¹

    drho_dx, drho_dy, _ = _density_gradients(grads, coords, bounds)

    u_g =  (G / (f * RHO0)) * drho_dy * depth_m        # m s⁻¹
    v_g = -(G / (f * RHO0)) * drho_dx * depth_m

    return u_g, v_g


def geostrophic_residual(
    grads: dict[str, Tensor],
    coords: Tensor,
    bounds: dict,
    ro_threshold: float = 0.1,
    depth_min: float = 50.0,
) -> Tensor:
    """Geostrophic balance loss [scalar].

    Penalises the magnitude of the diagnosed geostrophic velocity, masked to
    the QG regime (Ro < threshold) and below the Ekman layer (z > depth_min).

    The diagnosed u_g and v_g are computed from the baroclinic pressure.
    Under exact geostrophy f·v_g = (g/ρ₀)·∂ρ/∂x·z and f·u_g = −(g/ρ₀)·∂ρ/∂y·z
    by construction, so the *ageostrophic* residual is 0.  The loss therefore
    regularises the **amplitude** of geostrophic shear relative to f, preventing
    the network from producing unrealistically large density gradients.

    Parameters
    ----------
    ro_threshold:
        Rossby-number threshold above which the loss is suppressed.
        Default 0.1 (QG limit).
    depth_min:
        Minimum depth [m] for applying the constraint.  Default 50 m
        (below the Ekman layer).

    Returns
    -------
    loss : scalar Tensor
        Masked mean of (f·v_g − (g/ρ₀)·∂ρ/∂x·z)² + (f·u_g + (g/ρ₀)·∂ρ/∂y·z)²
        normalised by (f·U_typical)².
    """
    _, lat, depth_m, _ = _phys_coords(coords, bounds)
    f = coriolis(lat)
    # Compute density gradients once; inline geostrophic_velocity to avoid a
    # second _density_gradients (and _phys_coords) call inside that function.
    drho_dx, drho_dy, _ = _density_gradients(grads, coords, bounds)
    u_g =  (G / (f * RHO0)) * drho_dy * depth_m
    v_g = -(G / (f * RHO0)) * drho_dx * depth_m

    L_Ro  = 50e3
    U_typ = 0.5
    f_abs = f.abs()
    Ro    = (u_g.abs() + v_g.abs()) / (2.0 * f_abs * L_Ro)

    mask = (depth_m > depth_min) & (Ro < ro_threshold)
    m    = mask.float()
    loss = ((u_g * f_abs / U_typ).pow(2) + (v_g * f_abs / U_typ).pow(2)) * m
    return loss.sum() / m.sum().clamp(min=1.0)


def geostrophic_residual_full(
    grads: dict[str, Tensor],
    coords: Tensor,
    bounds: dict,
    u_net: Tensor,
    v_net: Tensor,
    ro_threshold: float = 0.1,
    depth_min: float = 50.0,
) -> Tensor:
    """Full geostrophic balance residual when the network outputs u and v.

    Directly checks  f·v − (1/ρ₀)·∂p/∂x = 0  and  f·u + (1/ρ₀)·∂p/∂y = 0
    using the network's velocity outputs and the baroclinic pressure gradient
    from the density field.

    Parameters
    ----------
    u_net, v_net : Tensor (B,)
        Eastward and northward velocities [m s⁻¹] from the network.

    Returns
    -------
    loss : scalar Tensor
        Masked MSE of the momentum residuals, normalised by (f·U_typ)².
    """
    _, lat, depth_m, _ = _phys_coords(coords, bounds)
    f = coriolis(lat)

    drho_dx, drho_dy, _ = _density_gradients(grads, coords, bounds)

    # Baroclinic pressure gradient [Pa m⁻¹] using the depth-linear approximation
    dPbc_dx = G * drho_dx * depth_m
    dPbc_dy = G * drho_dy * depth_m

    R_x = f * v_net - dPbc_dx / RHO0     # f·v − (1/ρ₀)∂p/∂x  [m s⁻²]
    R_y = f * u_net + dPbc_dy / RHO0     # f·u + (1/ρ₀)∂p/∂y

    U_typ = 0.5
    depth_mask = depth_m > depth_min

    Ro = (u_net.abs() + v_net.abs()) / (2.0 * f.abs() * 50e3)
    mask = depth_mask & (Ro < ro_threshold)

    if not mask.any():
        return grads['CT'].new_zeros(())

    f_m = f[mask].abs()
    norm = (f_m * U_typ).pow(2).clamp(min=1e-20)
    return ((R_x[mask].pow(2) + R_y[mask].pow(2)) / norm).mean()
