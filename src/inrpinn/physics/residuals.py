"""Physics residual aggregator — single entry point for the PINN loss.

Imports all 11 physics modules and exposes:
  compute_residuals(model_output, coords, config) -> dict[str, Tensor]

Keys match the physics_weights entries in the config YAML (eq1_eos … eq11_mixed_layer).
Each value is a scalar loss term (already reduced over the batch).

The PINN loss function in training/loss.py calls this function and multiplies
each residual by its corresponding weight λᵢ from the config.  Setting all λᵢ
to 0.0 produces the pure INR data loss, making the two models exactly comparable.
"""
