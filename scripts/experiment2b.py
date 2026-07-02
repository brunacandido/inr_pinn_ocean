#!/usr/bin/env python3
"""Experiment 2B — INR: dense training + disjoint spatial val/test squares.

Validation and test profiles each come from their own set of randomly placed
spatial squares, with no spatial overlap between the two sets or with training.
Default split: 90% train / 5% val / 5% test.
All shared logic lives in inrpinn.experiments.runner.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# BooleanOptionalAction was added in Python 3.9; polyfill for older runtimes
if not hasattr(argparse, "BooleanOptionalAction"):
    class _BoolOpt(argparse.Action):
        def __init__(self, option_strings, dest, default=True, **kw):
            opts = [o for o in option_strings if o.startswith("--")]
            neg  = ["--no-" + o[2:] for o in opts]
            super().__init__(opts + neg, dest, nargs=0, default=default, **kw)
        def __call__(self, parser, ns, values, opt=None):  # noqa: ARG002
            del parser, values  # unused; required by argparse Action interface
            setattr(ns, self.dest, not (opt or "").startswith("--no-"))
    argparse.BooleanOptionalAction = _BoolOpt  # type: ignore[attr-defined]

from inrpinn.experiments.runner import run_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experiment 2B: INR — dense training, disjoint spatial val/test squares.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--zarr-path",  type=Path,
                   default=PROJECT_ROOT / "data" / "glorys_patch_34S69E.zarr")
    p.add_argument("--zarr-group", type=str, default="raw")
    p.add_argument("--config",     type=Path,
                   default=PROJECT_ROOT / "configs" / "pinn_patch_34S69E.yaml")
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "results" / "experiment2b")
    p.add_argument("--var-temp",   type=str, default="thetao")
    p.add_argument("--var-sal",    type=str, default="so")
    p.add_argument("--label-temp", type=str, default="Temperature (°C)")
    p.add_argument("--label-sal",  type=str, default="Salinity (PSU)")

    p.add_argument("--train-fraction",   type=float, default=0.90,
                   help="Fraction of all profiles used for training.")
    p.add_argument("--val-fraction",     type=float, default=0.05,
                   help="Fraction of all profiles for validation (in spatial squares).")
    p.add_argument("--n-val-squares",    type=int,   default=3,
                   help="Number of spatial squares for validation.")
    p.add_argument("--n-test-squares",   type=int,   default=3,
                   help="Number of spatial squares for test (separate from val).")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--weekly-subsample", action="store_true", default=False,
                   help="Keep one random time step per week per (lon, lat) location.")
    p.add_argument("--data-fraction",    type=float, default=None,
                   help="Keep only this fraction of profiles (e.g. 0.1 = 10%%).")
    p.add_argument("--depth-fraction",   type=float, default=None,
                   help="Keep all surface obs + this fraction of deeper depth obs.")

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
    p.add_argument("--num-workers",      type=int,   default=8,
                   help="DataLoader worker processes (0 = main process only).")
    p.add_argument("--amp",              action=argparse.BooleanOptionalAction, default=True,
                   help="Use automatic mixed precision on CUDA (--no-amp to disable).")
    p.add_argument("--val-every",        type=int,   default=5,
                   help="Run validation every N epochs (1 = every epoch).")
    p.add_argument("--compile",          action=argparse.BooleanOptionalAction, default=False,
                   help="torch.compile the model for extra GPU speed (PyTorch >= 2.0).")
    p.add_argument("--profile-epochs",   type=int,   default=0,
                   help="Profile this many training epochs with line_profiler then exit "
                        "(0 = disabled, run full training).")
    p.add_argument("--profile-sync",     action="store_true", default=False,
                   help="Insert torch.cuda.synchronize() after each CUDA op so "
                        "line_profiler shows true GPU wall-clock time (slower).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        exp_name       = "Experiment 2B",
        val_mode       = "disjoint_squares",
        val_mode_label = (f"Disjoint Squares "
                          f"({args.n_val_squares} val + {args.n_test_squares} test)"),
        args           = args,
        extra_split_kwargs = {
            "n_val_squares":  args.n_val_squares,
            "n_test_squares": args.n_test_squares,
        },
    )


if __name__ == "__main__":
    main()
