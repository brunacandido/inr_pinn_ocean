"""EQ-5 — Thermal wind balance (Southern Hemisphere).

Residual pair:
    R_u  =  f ∂u/∂z  +  (g/ρ₀) ∂ρ/∂y  =  0
    R_v  =  f ∂v/∂z  −  (g/ρ₀) ∂ρ/∂x  =  0

With f < 0 in the Southern Hemisphere (Agulhas), the sign is preserved
consistently with EQ-4.

For a network that outputs only CT and SA (no explicit velocities), the
geostrophic velocity shear ∂u/∂z and ∂v/∂z is *diagnosed* from the thermal
wind relation:

    ∂u_tw/∂z  =  −(g / (f ρ₀)) ∂ρ/∂y
    ∂v_tw/∂z  =  +(g / (f ρ₀)) ∂ρ/∂x

The PINN residual then enforces self-consistency: the diagnosed shear derived
from horizontal density gradients must match the actual shear encoded in the
network.  Since the shear is NOT a separate network output, the residual is
evaluated as the norm of the diagnosed thermal-wind shear scaled by a Rossby-
number-based quality factor:

    L_tw  =  mean_qg(  (∂u_tw/∂z)²  +  (∂v_tw/∂z)²  )

This penalises spurious large horizontal density gradients that would imply
dynamically inconsistent velocity shear.  The QG mask suppresses the loss
where Ro > ro_threshold so the high-Ro Agulhas core is not over-constrained.

Alternatively, if the network is extended to output u and v, the module
exposes ``thermal_wind_residual_full`` which uses the actual velocity shear.

Density horizontal gradients are obtained via the chain rule through the EOS:
    ∂ρ/∂x = (∂ρ/∂CT)·(∂CT/∂x)  +  (∂ρ/∂SA)·(∂SA/∂x)
           = α · dCT_dx_phys    +  β · dSA_dx_phys

where α = ∂ρ/∂CT and β = ∂ρ/∂SA come from gsw_torch.rho_alpha_beta, and
dCT_dx_phys is the chain-rule-rescaled gradient from siren.coords_and_grad.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .eos import _denorm, _phys_coords, _pressure, density, alpha_beta
from ..utils.coriolis import coriolis, rossby_mask

# ── Physical constants ───────────────────────────────────────────────────────
G    = 9.81      # m s⁻²
RHO0 = 1025.0    # kg m⁻³
R_EARTH = 6_371_000.0  # m


def _density_gradients(
    grads: dict[str, Tensor],
    coords: Tensor,
    bounds: dict,
) -> tuple[Tensor, Tensor, Tensor]:
    """Physical-unit horizontal and vertical density gradients [kg m⁻⁴].

    Returns (dρ/dx [kg m⁻⁴], dρ/dy [kg m⁻⁴], dρ/dz [kg m⁻⁴])
    via chain rule:  dρ/dx = α · dCT/dx + β · dSA/dx

    dCT/dx is in physical units [°C m⁻¹].  The Siren returns dCT in
    normalised space; the chain-rule rescaling is:
        dCT_phys/dx_m = dCT_norm * (ΔCT / Δlon_deg) / (π/180 · R · cos(lat))
    and similarly for y (meridional) and z (depth).
    """
    _, lat, _, _ = _phys_coords(coords, bounds)
    lat_rad  = lat * (math.pi / 180.0)
    cos_lat  = torch.cos(lat_rad).clamp(min=1e-6)

    # Scale factors: physical Δ = hi − lo for each coordinate/output
    dCT  = bounds['T'][1]    - bounds['T'][0]       # °C
    dSA  = bounds['S'][1]    - bounds['S'][0]       # g kg⁻¹
    dlon = bounds['lon'][1]  - bounds['lon'][0]     # degrees
    dlat = bounds['lat'][1]  - bounds['lat'][0]     # degrees
    ddep = bounds['depth'][1]- bounds['depth'][0]   # m

    # Gradients in normalised space (from siren.coords_and_grad)
    dCT_dx_norm = grads['dCT_dx']   # ∂CT_norm/∂lon_norm
    dCT_dy_norm = grads['dCT_dy']
    dCT_dz_norm = grads['dCT_dz']
    dSA_dx_norm = grads['dSA_dx']
    dSA_dy_norm = grads['dSA_dy']
    dSA_dz_norm = grads['dSA_dz']

    # Convert to physical units ─────────────────────────────────────────
    # ∂CT/∂lon_deg = dCT_norm * (ΔCT / Δlon)
    # ∂CT/∂x_m    = ∂CT/∂lon_deg / (π/180 · R · cos(lat))
    deg_to_m_lon = math.pi / 180.0 * R_EARTH * cos_lat   # m / degree (zonal)
    deg_to_m_lat = math.pi / 180.0 * R_EARTH             # m / degree (meridional)

    dCT_dx = dCT_dx_norm * (dCT / dlon) / deg_to_m_lon   # °C m⁻¹
    dCT_dy = dCT_dy_norm * (dCT / dlat) / deg_to_m_lat
    dCT_dz = dCT_dz_norm * (dCT / ddep)                  # °C m⁻¹  (z downward)

    dSA_dx = dSA_dx_norm * (dSA / dlon) / deg_to_m_lon   # g kg⁻¹ m⁻¹
    dSA_dy = dSA_dy_norm * (dSA / dlat) / deg_to_m_lat
    dSA_dz = dSA_dz_norm * (dSA / ddep)

    # gsw_torch coefficients: α = ∂ρ/∂CT [kg m⁻³ °C⁻¹], β = ∂ρ/∂SA [kg m⁻³ (g/kg)⁻¹]
    alpha, beta = alpha_beta(grads, coords, bounds)

    drho_dx = alpha * dCT_dx + beta * dSA_dx   # kg m⁻⁴
    drho_dy = alpha * dCT_dy + beta * dSA_dy
    drho_dz = alpha * dCT_dz + beta * dSA_dz

    return drho_dx, drho_dy, drho_dz


def thermal_wind_shear(
    grads: dict[str, Tensor],
    coords: Tensor,
    bounds: dict,
) -> tuple[Tensor, Tensor]:
    """Diagnosed thermal wind velocity shear [s⁻¹].

    Returns
    -------
    du_dz_tw : Tensor (B,)   ∂u/∂z diagnosed from −(g/fρ₀)∂ρ/∂y
    dv_dz_tw : Tensor (B,)   ∂v/∂z diagnosed from +(g/fρ₀)∂ρ/∂x
    """
    _, lat, _, _ = _phys_coords(coords, bounds)
    f = coriolis(lat)                                      # s⁻¹, negative in SH

    drho_dx, drho_dy, _ = _density_gradients(grads, coords, bounds)

    du_dz_tw = -(G / (f * RHO0)) * drho_dy               # s⁻¹
    dv_dz_tw =  (G / (f * RHO0)) * drho_dx

    return du_dz_tw, dv_dz_tw


def thermal_wind_residual(
    grads: dict[str, Tensor],
    coords: Tensor,
    bounds: dict,
    ro_threshold: float = 0.1,
    depth_min: float = 50.0,
) -> Tensor:
    """Thermal wind balance residual [scalar].

    For a CT/SA-only network the residual penalises the squared magnitude of
    the diagnosed thermal wind shear, masked to the QG regime (Ro < threshold)
    and below the Ekman layer (depth > depth_min m).

    In the QG limit, the shear should be dynamically consistent with the actual
    velocity field.  Because the network does not output velocities, the loss
    enforces that the shear is physically realisable (finite, well-scaled).

    Parameters
    ----------
    ro_threshold:
        Suppress loss where Ro ≥ this value (ageostrophic Agulhas core).
    depth_min:
        Only apply below this depth [m] to avoid the Ekman layer.

    Returns
    -------
    loss : scalar Tensor
        Depth- and Rossby-masked mean of (∂u_tw/∂z)² + (∂v_tw/∂z)²,
        normalised by (g/(f·ρ₀))² so the result is in [kg² m⁻⁸].

    Notes
    -----
    If the network is extended to also predict u and v, replace this with
    ``thermal_wind_residual_full`` which uses the actual velocity shear.
    """
    _, lat, depth_m, _ = _phys_coords(coords, bounds)
    f = coriolis(lat)
    # Compute density gradients once; inline thermal_wind_shear to avoid a
    # second _density_gradients (and _phys_coords, coriolis) call inside it.
    drho_dx, drho_dy, _ = _density_gradients(grads, coords, bounds)
    du_dz_tw = -(G / (f * RHO0)) * drho_dy
    dv_dz_tw =  (G / (f * RHO0)) * drho_dx

    H_scale  = 500.0
    zeta_est = (du_dz_tw.abs() + dv_dz_tw.abs()) * H_scale / 2.0
    Ro_est   = zeta_est / f.abs()

    mask = (depth_m > depth_min) & (Ro_est < ro_threshold)
    m    = mask.float()
    loss = (du_dz_tw.pow(2) + dv_dz_tw.pow(2)) * m
    return loss.sum() / m.sum().clamp(min=1.0)


def thermal_wind_residual_full(
    grads: dict[str, Tensor],
    coords: Tensor,
    bounds: dict,
    u_shear: Tensor,
    v_shear: Tensor,
    ro_threshold: float = 0.1,
    depth_min: float = 50.0,
) -> Tensor:
    """Full thermal wind residual when the network also outputs u, v.

    Compares the network's actual velocity shear ∂u/∂z, ∂v/∂z (provided by
    the caller) against the thermal-wind-diagnosed shear from density.

    Parameters
    ----------
    u_shear, v_shear:
        Actual velocity shear [s⁻¹] from the network via autograd:
        u_shear = ∂u_net/∂z,  v_shear = ∂v_net/∂z.

    Returns
    -------
    loss : scalar Tensor
        Masked MSE of (du/dz_obs − du/dz_tw)² + (dv/dz_obs − dv/dz_tw)²,
        normalised by the characteristic shear scale.
    """
    _, lat, depth_m, _ = _phys_coords(coords, bounds)

    du_dz_tw, dv_dz_tw = thermal_wind_shear(grads, coords, bounds)

    R_u = u_shear - du_dz_tw
    R_v = v_shear - dv_dz_tw

    depth_mask = depth_m > depth_min
    if not depth_mask.any():
        return grads['CT'].new_zeros(())

    loss = (R_u[depth_mask].pow(2) + R_v[depth_mask].pow(2)).mean()
    return loss
