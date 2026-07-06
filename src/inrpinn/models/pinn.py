"""PINN model — physics-informed neural representation.

All 11 physics equations from `physics/` are available.  Total loss:

    L = L_data + Σ_k( w_k · L_k )

where k is any subset of the 11 equations.

Equation catalogue
------------------
  eq1_eos           — TEOS-10 EOS self-consistency
  eq2_hydrostatic   — ∂p/∂z = ρg
  eq3_continuity    — ∇·u = 0
  eq4_geostrophic   — geostrophic balance (Ro-masked)
  eq5_thermal_wind  — thermal-wind shear (Ro-masked)
  eq6_qgpv          — QG PV conservation proxy (Ro-masked)
  eq7_brunt_vaisala — N² ≥ 0 static stability (inequality)
  eq8_ertel_pv      — Ertel PV conservation (β-term)
  eq9_salinity_adv  — ∂SA/∂t + u·∇SA = κ_S ∂²SA/∂z²  (needs 2nd-order autograd)
  eq10_temp_adv     — ∂CT/∂t + u·∇CT = κ_T ∂²CT/∂z²  (needs 2nd-order autograd)
  eq11_mixed_layer  — near-surface |∂CT/∂t| soft bound

Selecting equations
-------------------
Pass ``active_eqs`` to restrict which equations are computed at all.
Default (None) computes every equation that appears in ``weights``.
Setting a weight to 0 keeps the equation in the log (for monitoring)
but does not add it to the total loss.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .siren import Siren
from ..physics.eos import eos_residual
from ..physics.hydrostatic import hydrostatic_residual
from ..physics.continuity import continuity_residual
from ..physics.geostrophic import geostrophic_residual
from ..physics.thermal_wind import thermal_wind_residual
from ..physics.qgpv import qgpv_residual
from ..physics.brunt_vaisala import brunt_vaisala_residual
from ..physics.ertel_pv import ertel_pv_residual
from ..physics.salinity_adv import salinity_adv_residual
from ..physics.temp_adv import temp_adv_residual
from ..physics.mixed_layer import mixed_layer_residual

# Ordered list of all supported equations (matches pinn_agulhas.yaml keys)
ALL_EQUATIONS: list[str] = [
    'eq1_eos',
    'eq2_hydrostatic',
    'eq3_continuity',
    'eq4_geostrophic',
    'eq5_thermal_wind',
    'eq6_qgpv',
    'eq7_brunt_vaisala',
    'eq8_ertel_pv',
    'eq9_salinity_adv',
    'eq10_temp_adv',
    'eq11_mixed_layer',
]


class PINN(nn.Module):
    """Physics-informed neural network for 4-D ocean CT / SA reconstruction.

    Parameters
    ----------
    siren : Siren
        Backbone — architecturally identical to the INR; INR weights can
        be used to warm-start training.
    weights : dict[str, float]
        Per-equation loss weight.  Missing keys default to 0 (not added to
        total, but still computed and logged if the equation is active).
    bounds : dict
        Normalisation bounds.  Required keys: lon, lat, depth, time, T, S.
    masking : dict, optional
        Keys used: ``depth_min_geostrophic`` [m], ``ro_threshold`` [–],
        ``depth_min_diffusion`` [m or null].
    diffusivity : dict, optional
        Keys: ``kappa_T`` [m²/s], ``kappa_S`` [m²/s].  Defaults to 1e-5.
    active_eqs : list[str] | None
        Equations to compute.  None (default) = all equations present in
        ``weights``.  Pass a subset to skip expensive equations entirely,
        e.g. ``['eq2_hydrostatic', 'eq3_continuity']``.
    """

    def __init__(
        self,
        siren: Siren,
        weights: dict[str, float],
        bounds: dict,
        masking: dict | None = None,
        diffusivity: dict | None = None,
        active_eqs: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.siren       = siren
        self.weights     = weights
        self.bounds      = bounds
        self.masking     = masking     or {}
        self.diffusivity = diffusivity or {}

        # Cache scalar masking/diffusivity params so _physics_loss avoids
        # repeated dict.get() + float() calls in the hot per-step loop.
        m = self.masking;  d = self.diffusivity
        self._ro  = float(m.get('ro_threshold', 0.1))
        self._dg  = float(m.get('depth_min_geostrophic') or 0.0)
        self._dif = float(m.get('depth_min_diffusion')   or 0.0)
        self._kT  = float(d.get('kappa_T', 1e-5))
        self._kS  = float(d.get('kappa_S', 1e-5))

        # Resolve active equation list
        if active_eqs is None:
            # Default: every equation mentioned in weights (any weight, including 0)
            self.active_eqs: list[str] = [
                eq for eq in ALL_EQUATIONS if eq in weights
            ]
        else:
            unknown = set(active_eqs) - set(ALL_EQUATIONS)
            if unknown:
                raise ValueError(f"Unknown equations: {unknown}.  Valid: {ALL_EQUATIONS}")
            self.active_eqs = list(active_eqs)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, coords: Tensor) -> dict[str, Tensor]:
        """Map normalised coordinates to normalised CT and SA.

        Parameters
        ----------
        coords : Tensor (B, 4) — [lon, lat, depth, time] each in [-1, 1].

        Returns
        -------
        dict : {"CT": Tensor (B,), "SA": Tensor (B,)}
        """
        out = self.siren(coords)
        return {"CT": out[:, 0], "SA": out[:, 1]}

    # ── Physics dispatch ──────────────────────────────────────────────────────

    def _physics_loss(self, key: str, grads: dict, colloc: Tensor) -> Tensor:
        """Compute a single physics residual by equation key."""
        b   = self.bounds
        ro  = self._ro
        dg  = self._dg
        dif = self._dif
        kT  = self._kT
        kS  = self._kS

        if key == 'eq1_eos':
            return eos_residual(grads, colloc, b)
        if key == 'eq2_hydrostatic':
            return hydrostatic_residual(grads, colloc, b)
        if key == 'eq3_continuity':
            return continuity_residual(grads, colloc, b, depth_min=dg)
        if key == 'eq4_geostrophic':
            return geostrophic_residual(grads, colloc, b, ro_threshold=ro, depth_min=dg)
        if key == 'eq5_thermal_wind':
            return thermal_wind_residual(grads, colloc, b, ro_threshold=ro, depth_min=dg)
        if key == 'eq6_qgpv':
            return qgpv_residual(grads, colloc, b, ro_threshold=ro, depth_min=dg)
        if key == 'eq7_brunt_vaisala':
            return brunt_vaisala_residual(grads, colloc, b)
        if key == 'eq8_ertel_pv':
            return ertel_pv_residual(grads, colloc, b, ro_threshold=ro, depth_min=dg)
        if key == 'eq9_salinity_adv':
            return salinity_adv_residual(grads, colloc, b, kappa_S=kS, depth_min=dif)
        if key == 'eq10_temp_adv':
            return temp_adv_residual(grads, colloc, b, kappa_T=kT, depth_min=dif)
        if key == 'eq11_mixed_layer':
            return mixed_layer_residual(grads, colloc, b)
        return grads['CT'].new_zeros(())

    # ── Loss ──────────────────────────────────────────────────────────────────

    def loss(
        self,
        coords: Tensor,
        targets: Tensor,
        colloc: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Total PINN loss with per-equation logging.

        Parameters
        ----------
        coords  : (B, 4) normalised observation coordinates.
        targets : (B, 2) normalised (CT, SA) ground truth.
                  Column 0 = CT, column 1 = SA.
        colloc  : (C, 4) collocation coordinates for physics residuals.
                  Use ``PINN.sample_collocation(n, device)`` to generate.

        Returns
        -------
        total : scalar Tensor — differentiable, for .backward().
        log   : dict[str, Tensor] — all components detached.
                Keys: total, data, data_CT, data_SA, + one per active equation.

        Notes
        -----
        ``colloc.requires_grad_(True)`` is called in-place before the autograd
        pass so that EQ-9 / EQ-10 can compute second-order depth derivatives.
        """
        # ── Data loss ─────────────────────────────────────────────────────────
        pred    = self(coords)
        ct_loss = F.mse_loss(pred["CT"], targets[:, 0])
        sa_loss = F.mse_loss(pred["SA"], targets[:, 1])
        total   = ct_loss + sa_loss

        log: dict[str, Tensor] = {
            "data_CT": ct_loss.detach(),
            "data_SA": sa_loss.detach(),
            "data":    total.detach(),
        }

        # ── Physics losses ────────────────────────────────────────────────────
        if self.active_eqs:
            # Enable requires_grad on colloc so second-order autograd (EQ-9, EQ-10)
            # can differentiate through the first-order gradients w.r.t. depth.
            colloc.requires_grad_(True)
            grads = self.siren.coords_and_grad(colloc)

            for key in self.active_eqs:
                l_k = self._physics_loss(key, grads, colloc)
                w_k = self.weights.get(key, 0.0)
                if w_k > 0:
                    total = total + w_k * l_k
                log[key] = l_k.detach()

        log["total"] = total.detach()
        return total, log

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def sample_collocation(n: int, device: torch.device) -> Tensor:
        """Sample *n* collocation points uniformly in [-1, 1]^4."""
        return torch.rand(n, 4, device=device) * 2.0 - 1.0

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: dict, active_eqs: list[str] | None = None) -> "PINN":
        """Instantiate from a YAML config (pinn_agulhas.yaml).

        Parameters
        ----------
        cfg        : parsed YAML dict.
        active_eqs : optional equation subset (default: all in weights).

        Example
        -------
        >>> cfg   = yaml.safe_load(open("configs/pinn_agulhas.yaml"))
        >>> model = PINN.from_config(cfg)
        >>> # Only hydrostatic + continuity:
        >>> light = PINN.from_config(cfg, active_eqs=['eq2_hydrostatic', 'eq3_continuity'])
        """
        return cls(
            siren       = Siren.from_config(cfg),
            weights     = cfg.get("physics_weights", {}),
            bounds      = cfg["normalisation"],
            masking     = cfg.get("masking", {}),
            diffusivity = cfg.get("diffusivity", {}),
            active_eqs  = active_eqs,
        )

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        active  = [eq for eq in self.active_eqs if self.weights.get(eq, 0) > 0]
        passive = [eq for eq in self.active_eqs if self.weights.get(eq, 0) == 0]
        return (
            f"PINN(\n"
            f"  siren={self.siren},\n"
            f"  weighted={active},\n"
            f"  logged_only={passive}\n"
            f")"
        )
