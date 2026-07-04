#!/usr/bin/env python3
"""Experiment 3 — INR on sparse training data with disjoint spatial val/test squares.

Trains the INR model with a reduced fraction of non-surface training observations
to study how reconstruction quality degrades with increasing sparsity.
The validation and test splits are identical to Experiment 2B (disjoint squares,
90/5/5 split, seed=42) so results are directly comparable.

Typical usage via the sparsity sweep script:
    bash scripts/run_experiment3.sh

Or a single run:
    uv run scripts/experiment3.py --train-depths-data-fraction 0.10
"""

from __future__ import annotations

import argparse
from pathlib import Path

if not hasattr(argparse, "BooleanOptionalAction"):
    class _BoolOpt(argparse.Action):
        def __init__(self, option_strings, dest, default=True, **kw):
            opts = [o for o in option_strings if o.startswith("--")]
            neg  = ["--no-" + o[2:] for o in opts]
            super().__init__(opts + neg, dest, nargs=0, default=default, **kw)
        def __call__(self, parser, ns, values, opt=None):  # noqa: ARG002
            del parser, values
            setattr(ns, self.dest, not (opt or "").startswith("--no-"))
    argparse.BooleanOptionalAction = _BoolOpt  # type: ignore[attr-defined]

from inrpinn.experiments.runner import run_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Experiment 3: INR on sparse training data — increasing sparsity levels "
            "with fixed disjoint spatial val/test squares (same split as Experiment 2B)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--zarr-path",  type=Path,
                   default=PROJECT_ROOT / "data" / "glorys_patch_34S69E.zarr")
    p.add_argument("--zarr-group", type=str, default="raw")
    p.add_argument("--config",     type=Path,
                   default=PROJECT_ROOT / "configs" / "pinn_patch_34S69E.yaml")
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "results" / "experiment3")
    p.add_argument("--var-temp",   type=str, default="thetao")
    p.add_argument("--var-sal",    type=str, default="so")
    p.add_argument("--label-temp", type=str, default="Temperature (°C)")
    p.add_argument("--label-sal",  type=str, default="Salinity (PSU)")

    # ── Split (identical defaults to Experiment 2B for comparability) ──────────
    p.add_argument("--train-fraction",   type=float, default=0.90,
                   help="Fraction of all profiles for training.")
    p.add_argument("--val-fraction",     type=float, default=0.05,
                   help="Fraction of all profiles for validation (disjoint squares).")
    p.add_argument("--n-val-squares",    type=int,   default=3,
                   help="Number of spatial squares for validation.")
    p.add_argument("--n-test-squares",   type=int,   default=3,
                   help="Number of spatial squares for test (separate from val).")
    p.add_argument("--seed",             type=int,   default=42,
                   help="Random seed — keep at 42 to match Experiment 2B split.")

    # ── Sparsity ───────────────────────────────────────────────────────────────
    p.add_argument("--train-depths-data-fraction", type=float, default=None,
                   help=(
                       "Fraction of non-surface TRAINING observations to keep. "
                       "Surface training obs and all val/test obs are always kept. "
                       "None = keep all (equivalent to Experiment 2B)."
                   ))

    p.add_argument("--weekly-subsample", action="store_true", default=False,
                   help="Keep one random time step per week per location.")

    # ── Training ───────────────────────────────────────────────────────────────
    p.add_argument("--epochs",           type=int,   default=2000)
    p.add_argument("--batch-size",       type=int,   default=8192)
    p.add_argument("--lr",               type=float, default=None)
    p.add_argument("--patience",         type=int,   default=100)
    p.add_argument("--min-delta",        type=float, default=1e-6)
    p.add_argument("--checkpoint-every", type=int,   default=50)
    p.add_argument("--infer-batch",      type=int,   default=32768)
    p.add_argument("--device",           type=str,   default=None)
    p.add_argument("--resume",           type=Path,  default=None,
                   help="Path to a checkpoint to resume training from.")
    p.add_argument("--num-workers",      type=int,   default=8)
    p.add_argument("--amp",              action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--val-every",        type=int,   default=5)
    p.add_argument("--compile",          action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frac = args.train_depths_data_fraction
    sparsity_label = (
        f"sparsity {1 - frac:.0%}" if frac is not None else "full training data"
    )
    suffix = f"sparsity{int((1 - frac) * 100)}pct" if frac is not None else "full"
    run_experiment(
        exp_name       = "Experiment 3",
        val_mode       = "disjoint_squares",
        val_mode_label = (
            f"Disjoint Squares ({args.n_val_squares} val + {args.n_test_squares} test)"
            f" · {sparsity_label}"
        ),
        args           = args,
        extra_split_kwargs = {
            "n_val_squares":  args.n_val_squares,
            "n_test_squares": args.n_test_squares,
        },
        run_suffix = suffix,
    )


if __name__ == "__main__":
    main()
