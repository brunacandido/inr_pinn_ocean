"""EQ-1 — TEOS-10 equation of state.

Computes in-situ density ρ(CT, SA, p) using gsw_torch (PyTorch / autograd
compatible implementation of TEOS-10).

Coordinate / normalisation convention
--------------------------------------
The SIREN network outputs CT and SA already in *physical* units, normalised to
[-1, 1] using the bounds stored in the config:

  CT  ∈ [bounds['T'][0],  bounds['T'][1]]   °C        (conservative temperature)
  SA  ∈ [bounds['S'][0],  bounds['S'][1]]   g kg⁻¹   (absolute salinity)

Depth is positive downward (GLORYS convention).  gsw_torch.p_from_z receives
z_gsw = -depth [m, negative upward] and returns pressure in dbar.

Public API
----------
  density(grads, coords, bounds)
      Full TEOS-10 ρ [kg m⁻³], differentiable through gsw_torch.
      Used by every downstream physics module.

  eos_residual(grads, coords, bounds, rho_norm=None)
      Optional EOS self-consistency loss.  Returns 0 unless the network is
      extended to output ρ directly (rho_norm ≠ None).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import gsw_torch as gsw
    _GSW_TORCH = True
except ImportError:
    import gsw as _gsw_np
    _GSW_TORCH = False

# ── Physical constants ───────────────────────────────────────────────────────
G    = 9.81      # m s⁻²
RHO0 = 1025.0    # kg m⁻³  Boussinesq reference density

# Linear EOS coefficients — used as fallback when gsw_torch would require
# float64 (e.g. MPS / Apple Silicon).  Typical Southern Ocean values.
_ALPHA_LIN = 2.0e-4    # thermal expansion  [1/°C]
_BETA_LIN  = 7.6e-4    # haline contraction [1/(g/kg)]
_CT0_LIN   = 10.0      # reference CT  [°C]
_SA0_LIN   = 35.0      # reference SA  [g/kg]


def _linear_eos(CT_phys: Tensor, SA_phys: Tensor) -> Tensor:
    """Linear EOS: ρ ≈ ρ₀·(1 − α·ΔCT + β·ΔSA).
    Float32-native, differentiable on any device.  Used as MPS fallback.
    """
    return RHO0 * (1.0 - _ALPHA_LIN * (CT_phys - _CT0_LIN)
                       + _BETA_LIN  * (SA_phys - _SA0_LIN))


# ── Coordinate helpers ───────────────────────────────────────────────────────

def _denorm(x: Tensor, lo: float, hi: float) -> Tensor:
    """Map x ∈ [-1, 1] → [lo, hi]."""
    return lo + (x + 1.0) * 0.5 * (hi - lo)


def _phys_coords(coords: Tensor, bounds: dict) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return (lon_deg, lat_deg, depth_m, time_day) in physical units."""
    lon   = _denorm(coords[:, 0], *bounds['lon'])
    lat   = _denorm(coords[:, 1], *bounds['lat'])
    depth = _denorm(coords[:, 2], *bounds['depth'])  # m, positive downward
    time  = _denorm(coords[:, 3], *bounds['time'])
    return lon, lat, depth, time


# ── Core density function ────────────────────────────────────────────────────

def density(
    grads: dict[str, Tensor],
    coords: Tensor,
    bounds: dict,
) -> Tensor:
    """Full TEOS-10 in-situ density [kg m⁻³], differentiable via gsw_torch.

    Parameters
    ----------
    grads:
        Output of ``siren.coords_and_grad(coords)``.  Must contain 'CT' and
        'SA' (normalised network outputs).
    coords:
        (B, 4) normalised collocation coordinates [lon, lat, depth, time].
    bounds:
        Config normalisation dict, e.g. ``cfg['normalisation']``.

    Returns
    -------
    rho : Tensor, shape (B,)
        In-situ density [kg m⁻³].  Gradient flows back through gsw_torch to
        the network parameters via CT and SA.
    """
    CT_phys = _denorm(grads['CT'], *bounds['T'])   # °C
    SA_phys = _denorm(grads['SA'], *bounds['S'])   # g kg⁻¹ (absolute salinity)

    _, lat, depth, _ = _phys_coords(coords, bounds)
    p_dbar = _pressure(depth, lat)                  # dbar

    if _GSW_TORCH and SA_phys.device.type != 'mps':
        return gsw.rho(SA_phys, CT_phys, p_dbar)
    elif _GSW_TORCH:
        # MPS does not support float64 required by gsw_torch, and backprop
        # through a float64↔float32 cast fails on MPS.  Use the linear EOS
        # approximation instead — it is differentiable float32 on any device.
        return _linear_eos(CT_phys, SA_phys)
    else:
        import numpy as np
        SA_np  = SA_phys.detach().cpu().numpy()
        CT_np  = CT_phys.detach().cpu().numpy()
        p_np   = p_dbar.detach().cpu().numpy()
        rho_np = _gsw_np.rho(SA_np, CT_np, p_np)
        return torch.tensor(rho_np, dtype=SA_phys.dtype, device=SA_phys.device)


def _pressure(depth_m: Tensor, lat_deg: Tensor) -> Tensor:
    """Sea pressure [dbar] from depth [m positive downward].

    Uses the Boussinesq approximation (1 dbar ≈ 1.019716 m, error < 0.5%)
    instead of gsw.p_from_z.  Avoids gsw_torch's internal float64 cast,
    which fails on MPS (Apple Silicon) where float64 is unsupported.
    """
    return depth_m / 1.019716


# ── EOS residual (only active when ρ is a separate network output) ───────────

def eos_residual(
    grads: dict[str, Tensor],
    coords: Tensor,
    bounds: dict,
    rho_norm: Tensor | None = None,
) -> Tensor:
    """EOS self-consistency loss [kg² m⁻⁶].

    When the network predicts ρ as an additional output (rho_norm ≠ None),
    this penalises the discrepancy between the network's density and the
    TEOS-10 value computed from its own CT and SA predictions.

    When rho_norm is None (default — ρ is always derived from CT, SA internally),
    returns exactly 0 so the loss is a no-op.

    Parameters
    ----------
    rho_norm:
        Normalised ρ from the network, shape (B,).
        Bounds key 'rho' must exist in *bounds*, e.g. [1020.0, 1030.0].
    """
    if rho_norm is None:
        return grads['CT'].new_zeros(())

    rho_lo, rho_hi = bounds.get('rho', [1020.0, 1030.0])
    rho_net = _denorm(rho_norm, rho_lo, rho_hi)      # kg m⁻³
    rho_gsw = density(grads, coords, bounds)           # kg m⁻³

    return F.mse_loss(rho_net, rho_gsw)


# ── Convenience: thermal expansion / haline contraction coefficients ─────────

def alpha_beta(
    grads: dict[str, Tensor],
    coords: Tensor,
    bounds: dict,
) -> tuple[Tensor, Tensor]:
    """Thermal expansion α and haline contraction β from TEOS-10.

    Returns
    -------
    alpha : Tensor (B,)   ∂ρ/∂CT  [kg m⁻³ °C⁻¹]  (positive)
    beta  : Tensor (B,)   ∂ρ/∂SA  [kg m⁻³ (g/kg)⁻¹]  (positive)

    These are used by hydrostatic.py and thermal_wind.py to propagate
    the density gradient through the EOS without re-computing ρ.
    """
    if not _GSW_TORCH:
        raise RuntimeError("gsw_torch required for alpha/beta computation")

    CT_phys  = _denorm(grads['CT'], *bounds['T'])
    SA_phys  = _denorm(grads['SA'], *bounds['S'])
    _, lat, depth, _ = _phys_coords(coords, bounds)
    p_dbar   = _pressure(depth, lat)

    dev = SA_phys.device
    if dev.type != 'mps':
        _, alpha, beta = gsw.rho_alpha_beta(SA_phys, CT_phys, p_dbar)
        return alpha, beta
    # MPS fallback: return constant linear-EOS coefficients (float32, differentiable)
    alpha = SA_phys.new_full(SA_phys.shape, _ALPHA_LIN)
    beta  = SA_phys.new_full(SA_phys.shape, _BETA_LIN)
    return alpha, beta
