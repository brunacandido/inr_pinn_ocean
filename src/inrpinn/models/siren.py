"""SIREN — Sinusoidal Representation Network backbone (Sitzmann et al. 2020).

Architecture
------------
  SirenLayer(is_first=True)    : Linear(4 → H)   → sin(ω₀ · x)
  SirenLayer(is_first=False)×N : Linear(H → H)   → sin(ω₀ · x)
  nn.Linear                    : Linear(H → 2)   (no activation)

Outputs are conservative temperature CT and absolute salinity SA,
both in normalised [-1, 1] space.

Coordinate convention (input column indices):
  0 : lon   (x, zonal,       positive east)
  1 : lat   (y, meridional,  negative in Southern Hemisphere)
  2 : depth (z, positive downward, 0 at surface)
  3 : time  (t, day-of-year index, normalised)

Southern Hemisphere sign convention
------------------------------------
f = -2Ω sin(lat) < 0 for lat < 0.  No sign is hardcoded here; the sign of
lat (and therefore f) propagates naturally through physics modules that call
utils.coriolis.  coords[:, 1] carries the raw (negative) normalised latitude.

Derivative convention
----------------------
coords_and_grad() returns derivatives in *normalised coordinate space*.
To recover physical-unit gradients the caller must rescale:

  ∂CT_phys / ∂z_phys = (ΔCT_phys / ΔCT_norm) · (∂CT_norm / ∂z_norm) · (Δz_norm / Δz_phys)

where Δ·_phys = max − min from the config normalisation bounds and
Δ·_norm = 2 (the range of [-1, 1]).  This rescaling is done in each
physics module, not here, so the network stays free of domain constants.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class SirenLayer(nn.Module):
    """Single SIREN hidden layer: sin(ω₀ · (Wx + b))."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float,
        is_first: bool = False,
    ) -> None:
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self) -> None:
        n_in = self.linear.in_features
        with torch.no_grad():
            if self.is_first:
                # Ensures pre-activation inputs span roughly [-1, 1] so that
                # sin operates across its full range from the first forward pass.
                w_bound = 1.0 / n_in
            else:
                # Derived in Sitzmann et al. Supplementary Sec. 1.5 to
                # preserve the distribution of activations across layers.
                w_bound = math.sqrt(6.0 / n_in) / self.omega_0

            self.linear.weight.uniform_(-w_bound, w_bound)
            # Bias init follows PyTorch default: U(-1/√n_in, 1/√n_in).
            b_bound = 1.0 / math.sqrt(n_in)
            self.linear.bias.uniform_(-b_bound, b_bound)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class Siren(nn.Module):
    """SIREN network mapping (lon, lat, depth, time) → (CT, SA).

    Parameters
    ----------
    in_features:
        Number of input coordinates. Always 4 for this project
        (lon, lat, depth, time), but kept configurable for testing.
    out_features:
        Number of output fields. 2 for (CT, SA).
    hidden_dim:
        Width of every hidden SIREN layer.
    n_layers:
        Number of sine-activated hidden layers (excluding the final
        linear output layer).  n_layers=5 gives 5 sine layers + 1 linear.
    omega_0:
        Sinusoidal frequency multiplier applied in every layer.
        Sitzmann et al. recommend 30 for spatial coordinates.
    """

    def __init__(
        self,
        in_features: int = 4,
        out_features: int = 2,
        hidden_dim: int = 256,
        n_layers: int = 5,
        omega_0: float = 20.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega_0 = omega_0

        layers: list[nn.Module] = [
            SirenLayer(in_features, hidden_dim, omega_0, is_first=True)
        ]
        for _ in range(n_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim, omega_0, is_first=False))
        self.hidden = nn.Sequential(*layers)

        # No activation on the output layer — CT and SA are unbounded in
        # normalised space and the linear layer can represent any range.
        self.output_layer = nn.Linear(hidden_dim, out_features)
        self._init_output_layer()

    def _init_output_layer(self) -> None:
        n_in = self.output_layer.in_features
        with torch.no_grad():
            bound = math.sqrt(6.0 / n_in) / self.omega_0
            self.output_layer.weight.uniform_(-bound, bound)
            b_bound = 1.0 / math.sqrt(n_in)
            self.output_layer.bias.uniform_(-b_bound, b_bound)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, coords: Tensor) -> Tensor:
        """Map normalised coordinates to normalised (CT, SA).

        Parameters
        ----------
        coords:
            Shape (B, 4).  Columns: [lon, lat, depth, time], each in [-1, 1].
            If coords.requires_grad is True the computation graph is retained
            automatically; for derivative computation use coords_and_grad().

        Returns
        -------
        Tensor, shape (B, 2).
            Column 0: CT (normalised conservative temperature).
            Column 1: SA (normalised absolute salinity).
        """
        return self.output_layer(self.hidden(coords))

    # ------------------------------------------------------------------
    # Derivatives
    # ------------------------------------------------------------------

    def coords_and_grad(self, coords: Tensor) -> dict[str, Tensor]:
        """Forward pass + first-order spatial/temporal derivatives via autograd.

        Parameters
        ----------
        coords:
            Shape (B, 4) — normalised (lon, lat, depth, time).
            requires_grad is enabled in-place if not already set.

        Returns
        -------
        dict with keys:

          CT, SA            — shape (B,), normalised network outputs
          dCT_dx, dCT_dy,
          dCT_dz, dCT_dt   — ∂CT/∂lon, ∂CT/∂lat, ∂CT/∂depth, ∂CT/∂time
          dSA_dx, dSA_dy,
          dSA_dz, dSA_dt   — ∂SA/∂lon, ∂SA/∂lat, ∂SA/∂depth, ∂SA/∂time

        All derivatives are in normalised coordinate space.  Physics modules
        must apply the (Δphys / Δnorm) chain-rule factor to obtain physical
        units (see module docstring).

        create_graph=True is set on both grad calls so that the returned
        gradient tensors remain part of the computation graph.  This means:
          - Physics losses can be backpropagated through normally.
          - Higher-order derivatives (e.g. ∂²CT/∂z² for EQ-10) can be
            computed by calling autograd.grad on the returned dCT_dz tensor.

        Notes
        -----
        Using .sum() to reduce before autograd is valid because each output
        element out[i] depends only on coords[i] (no cross-sample coupling
        inside the batch), so d(Σᵢ out_i) / d(coords_j) = d(out_j) / d(coords_j).
        """
        # Enable gradient tracking on coords (no-op if already True).
        # requires_grad_() is in-place on leaf tensors; if coords is a
        # non-leaf (e.g. created by an operation), detach first.
        if not coords.requires_grad:
            coords = coords.detach().requires_grad_(True)

        out = self.forward(coords)  # (B, 2)
        CT = out[:, 0]              # (B,)
        SA = out[:, 1]              # (B,)

        # First call retains the forward graph for the second grad call.
        # create_graph=True implies retain_graph=True, but we set it
        # explicitly on the first call for clarity.
        grad_CT = torch.autograd.grad(
            CT.sum(),
            coords,
            create_graph=True,
            retain_graph=True,
        )[0]  # (B, 4): columns → [dCT/dx, dCT/dy, dCT/dz, dCT/dt]

        grad_SA = torch.autograd.grad(
            SA.sum(),
            coords,
            create_graph=True,
            retain_graph=True,
        )[0]  # (B, 4): columns → [dSA/dx, dSA/dy, dSA/dz, dSA/dt]

        return {
            # Outputs
            "CT":     CT,
            "SA":     SA,
            # CT gradients (dCT/d·  in normalised space)
            "dCT_dx": grad_CT[:, 0],   # ∂CT/∂lon
            "dCT_dy": grad_CT[:, 1],   # ∂CT/∂lat  (negative lat in SH)
            "dCT_dz": grad_CT[:, 2],   # ∂CT/∂depth
            "dCT_dt": grad_CT[:, 3],   # ∂CT/∂time  (used by EQ-10)
            # SA gradients
            "dSA_dx": grad_SA[:, 0],   # ∂SA/∂lon
            "dSA_dy": grad_SA[:, 1],   # ∂SA/∂lat
            "dSA_dz": grad_SA[:, 2],   # ∂SA/∂depth
            "dSA_dt": grad_SA[:, 3],   # ∂SA/∂time  (used by EQ-9)
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict) -> "Siren":
        """Instantiate from the `model.architecture` section of a YAML config.

        Example
        -------
        >>> import yaml
        >>> cfg = yaml.safe_load(open("configs/pinn_agulhas.yaml"))
        >>> model = Siren.from_config(cfg)
        """
        arch = cfg["model"]["architecture"]
        return cls(
            in_features=4,
            out_features=arch.get("out_features", 2),
            hidden_dim=arch["hidden_dim"],
            n_layers=arch["n_layers"],
            omega_0=arch["omega_0"],
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def n_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        n_hidden = len(self.hidden)
        return (
            f"Siren("
            f"in={self.in_features}, "
            f"out={self.out_features}, "
            f"hidden_dim={self.hidden[0].linear.out_features}, "
            f"n_layers={n_hidden}, "
            f"omega_0={self.omega_0}, "
            f"params={self.n_parameters():,})"
        )
