"""EQ-2 — Hydrostatic balance.

Residual:  R = ∂p/∂z − ρ g,  normalised by ρ₀ g

With z positive downward (GLORYS convention), pressure increases with depth so
∂p/∂z > 0.  The hydrostatic equation is:

    ∂p/∂z = ρ g

where p is the sea pressure at that depth.  The pressure is *not* a network
output; it is fixed by the depth coordinate via gsw_torch.p_from_z.  Hence
∂p/∂z is a known function of (z, lat) — only ρ = ρ(CT_net, SA_net, p) depends
on the network parameters.

The residual is:

    R = ( dp_coord/dz − ρ_net · g ) / ( ρ₀ · g )

which is dimensionless and ~0 when the network density matches the
hydrostatically consistent density at that depth.  Gradients flow back
through ρ_net → gsw_torch → CT_net, SA_net → SIREN weights.

dp_coord/dz is evaluated by a 1-m central finite difference on p_from_z
(no graph required — the coordinate is fixed, not a network output).
"""
from __future__ import annotations

import torch
from torch import Tensor

from .eos import _denorm, _phys_coords, _pressure, density

# ── Physical constants ───────────────────────────────────────────────────────
G    = 9.81     # m s⁻²
RHO0 = 1025.0   # kg m⁻³


def hydrostatic_residual(
    grads: dict[str, Tensor],
    coords: Tensor,
    bounds: dict,
) -> Tensor:
    """Normalised hydrostatic balance residual [dimensionless, scalar].

    Parameters
    ----------
    grads:
        Output of ``siren.coords_and_grad(coords)``.  Used for CT and SA.
    coords:
        (B, 4) normalised [lon, lat, depth, time].
    bounds:
        Config normalisation dict.

    Returns
    -------
    loss : scalar Tensor
        Mean-squared normalised residual  E[ ((∂p/∂z − ρg) / (ρ₀g))² ].
        Target value in the data: < 1e-4² based on GLORYS diagnostics (PASS).
    """
    _, lat, depth_m, _ = _phys_coords(coords, bounds)

    # ── dp/dz: finite-difference on the coordinate pressure ─────────────────
    # p_from_z is not a network output → compute without building a graph.
    half_dz = 0.5  # metres  (1-m central difference)
    with torch.no_grad():
        p_lo = _pressure(depth_m - half_dz, lat)   # dbar  (shallower)
        p_hi = _pressure(depth_m + half_dz, lat)   # dbar  (deeper)
    # Convert dbar → Pa (1 dbar = 1e4 Pa), divide by 1 m step
    dp_dz = (p_hi - p_lo).detach() * 1e4           # Pa m⁻¹, no gradient

    # ── ρ from network CT and SA (has gradient) ──────────────────────────────
    rho = density(grads, coords, bounds)             # kg m⁻³

    # ── Normalised residual ──────────────────────────────────────────────────
    R = (dp_dz - rho * G) / (RHO0 * G)             # dimensionless

    return R.pow(2).mean()
