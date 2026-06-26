"""INR model — pure implicit neural representation (data loss only).

Wraps the Siren backbone and exposes:
  forward(coords)          -> {"CT": ..., "SA": ...}   (normalised, shape (B,) each)
  loss(coords, targets)    -> (total_loss, {"CT": ct_loss, "SA": sa_loss})

CT and SA losses are kept separate so training can monitor which variable
is harder to reconstruct, without any extra cost.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .siren import Siren


class INR(nn.Module):
    """Implicit Neural Representation wrapping a SIREN backbone.

    Parameters
    ----------
    siren : Siren
        Backbone network mapping (B, 4) coords → (B, 2) outputs.
        Output column 0 = CT, column 1 = SA.
    """

    def __init__(self, siren: Siren) -> None:
        super().__init__()
        self.siren = siren

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, coords: Tensor) -> dict[str, Tensor]:
        """Map normalised coordinates to normalised CT and SA.

        Parameters
        ----------
        coords : Tensor, shape (B, 4)
            Normalised (lon, lat, depth, time) in [-1, 1].

        Returns
        -------
        dict
            "CT" : Tensor (B,) — normalised conservative temperature
            "SA" : Tensor (B,) — normalised absolute salinity
        """
        out = self.siren(coords)   # (B, 2)
        return {"CT": out[:, 0], "SA": out[:, 1]}

    # ── Loss ──────────────────────────────────────────────────────────────────

    def loss(
        self, coords: Tensor, targets: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Data loss: MSE on CT and SA, computed and returned separately.

        Parameters
        ----------
        coords  : Tensor, shape (B, 4) — normalised coordinates.
        targets : Tensor, shape (B, 2) — normalised (CT, SA) ground truth.
                  Column 0 = CT, column 1 = SA.

        Returns
        -------
        total_loss : Tensor scalar  — CT_loss + SA_loss (differentiable).
        per_var    : dict {"CT": Tensor scalar, "SA": Tensor scalar}
                     Individual losses for tracking; both are part of the
                     computation graph so callers can inspect or log them
                     without detaching.
        """
        pred    = self(coords)
        ct_loss = F.mse_loss(pred["CT"], targets[:, 0])
        sa_loss = F.mse_loss(pred["SA"], targets[:, 1])
        return ct_loss + sa_loss, {"CT": ct_loss, "SA": sa_loss}

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: dict) -> "INR":
        """Instantiate from the model section of a YAML config.

        Example
        -------
        >>> import yaml
        >>> cfg = yaml.safe_load(open("configs/inr_agulhas.yaml"))
        >>> model = INR.from_config(cfg)
        """
        return cls(siren=Siren.from_config(cfg))

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return f"INR(siren={self.siren})"
