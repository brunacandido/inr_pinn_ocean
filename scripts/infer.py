#!/usr/bin/env python3
"""Run inference on a saved checkpoint — works during or after training.

Usage
-----
    # Mid-training: point at any periodic checkpoint
    python scripts/infer.py \
        --checkpoint results/experiment2/20240101_120000/checkpoints/epoch_0500.pt \
        --zarr-path  data/glorys_patch_34S69E.zarr \
        --config     configs/pinn_patch_34S69E.yaml \
        --val-mode   contiguous \
        --seed       42

    # After training: use best_model.pt (no extra args needed)
    python scripts/infer.py \
        --checkpoint results/experiment2/20240101_120000/best_model.pt \
        --zarr-path  data/glorys_patch_34S69E.zarr \
        --config     configs/pinn_patch_34S69E.yaml \
        --val-mode   contiguous \
        --seed       42
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import yaml  # noqa: F401 — used via yaml.safe_load

from inrpinn.data.splitter import GlorysProfileSplitter
from inrpinn.models.inr import INR
from inrpinn.experiments.runner import (
    get_device, load_dataset, build_arrays,
    infer_split, compute_metrics, _subsample_depths, _compact_result,
)
from inrpinn.experiments.plotting import (
    build_pdf,
    pdf_spatial_fields, pdf_temporal_profiles,
    pdf_split_map, pdf_training, pdf_evaluation, pdf_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference-only evaluation on a saved checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to a .pt checkpoint file (best_model.pt or any epoch_XXXX.pt).")
    p.add_argument("--zarr-path",  type=Path,
                   default=PROJECT_ROOT / "data" / "glorys_patch_34S69E.zarr")
    p.add_argument("--config",     type=Path,
                   default=PROJECT_ROOT / "configs" / "pinn_patch_34S69E.yaml")

    # Split parameters (must match the original training run)
    p.add_argument("--val-mode",        type=str,   default="uniform",
                   choices=["uniform", "contiguous"])
    p.add_argument("--train-fraction",  type=float, default=0.70)
    p.add_argument("--val-fraction",    type=float, default=0.15)
    p.add_argument("--n-val-squares",   type=int,   default=5)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--weekly-subsample", action="store_true", default=False)
    p.add_argument("--data-fraction",   type=float, default=None)
    p.add_argument("--depth-fraction",  type=float, default=None)
    p.add_argument("--var-temp",        type=str,   default="thetao")
    p.add_argument("--var-sal",         type=str,   default="so")
    p.add_argument("--label-temp",      type=str,   default="Temperature (°C)")
    p.add_argument("--label-sal",       type=str,   default="Salinity (PSU)")
    p.add_argument("--zarr-group",      type=str,   default="raw")

    # Inference settings
    p.add_argument("--device",      type=str, default=None)
    p.add_argument("--infer-batch", type=int, default=131072)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-pdf",      action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_path = args.checkpoint.resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt   = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg    = ckpt.get("cfg") or yaml.safe_load(open(args.config))
    device = get_device(args.device)
    bounds = cfg["normalisation"]

    model = INR.from_config(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Epoch {ckpt.get('epoch', '?')}  |  "
          f"val_loss {ckpt.get('val_loss', float('nan')):.6f}  |  "
          f"params {model.n_parameters():,}")

    # ── Dataset + split ───────────────────────────────────────────────────────
    print(f"\nLoading dataset …")
    ds       = load_dataset(args.zarr_path, args.zarr_group)
    depths_g = ds.depth.values
    print(f"  {dict(ds.sizes)}")

    print(f"\nRebuilding {args.val_mode} split (seed={args.seed}) …")
    splitter = GlorysProfileSplitter(ds, variables=(args.var_temp, args.var_sal))
    split_kw = dict(
        mode             = args.val_mode,
        train_fraction   = args.train_fraction,
        val_fraction     = args.val_fraction,
        seed             = args.seed,
        weekly_subsample = args.weekly_subsample,
        data_fraction    = args.data_fraction,
    )
    if args.val_mode == "contiguous":
        split_kw["n_val_squares"] = args.n_val_squares

    result = splitter.split(**split_kw)
    splitter.print_summary(result)
    result = _compact_result(result)

    print("\nBuilding arrays …", flush=True)
    coords_np, targets_np, targets_phys = build_arrays(
        result, ds, args.var_temp, args.var_sal, bounds,
    )

    if args.depth_fraction is not None:
        result, coords_np, targets_np, targets_phys = _subsample_depths(
            result, coords_np, targets_np, targets_phys,
            args.depth_fraction, args.seed,
        )
        print(f"  Depth subsample ({args.depth_fraction:.0%}): {len(coords_np):,} obs")

    num_workers = args.num_workers
    use_amp     = device.type == "cuda"

    # ── Inference ─────────────────────────────────────────────────────────────
    print("\nEvaluating validation set …", flush=True)
    T_vp, S_vp = infer_split(model, coords_np, result.val_mask, device,
                              args.infer_batch, num_workers, use_amp)
    val_m = compute_metrics(T_vp, S_vp,
                            targets_phys[result.val_mask],
                            result.i_dep[result.val_mask], depths_g, bounds)
    print(f"  T RMSE: {val_m['T_rmse']:.4f}°C   S RMSE: {val_m['S_rmse']:.5f} PSU")

    print("Evaluating test set …", flush=True)
    T_tp, S_tp = infer_split(model, coords_np, result.test_mask, device,
                              args.infer_batch, num_workers, use_amp)
    test_m = compute_metrics(T_tp, S_tp,
                             targets_phys[result.test_mask],
                             result.i_dep[result.test_mask], depths_g, bounds)
    print(f"  T RMSE: {test_m['T_rmse']:.4f}°C   S RMSE: {test_m['S_rmse']:.5f} PSU")

    # ── Report ────────────────────────────────────────────────────────────────
    if not args.no_pdf:
        out_dir  = ckpt_path.parent
        epoch    = ckpt.get("epoch", 0)
        history  = ckpt.get("history")

        # Attach summary keys so pdf_summary and pdf_training work correctly
        if history is not None:
            history["best_epoch"]    = history.get("best_epoch", epoch)
            history["best_val_loss"] = ckpt.get("best_val", ckpt.get("val_loss", float("nan")))
        summary_history = history or {"best_epoch": epoch,
                                      "best_val_loss": ckpt.get("val_loss", float("nan"))}

        print("\nBuilding PDF …", flush=True)
        pages = [
            pdf_spatial_fields(ds, args.var_temp, args.var_sal,
                               args.label_temp, args.label_sal),
            pdf_temporal_profiles(ds, result, depths_g,
                                  args.var_temp, args.var_sal,
                                  args.label_temp, args.label_sal),
            pdf_split_map(result, ds),
        ]
        if history is not None:
            pages.append(pdf_training(history))
        pages += [
            pdf_evaluation("Validation", val_m, depths_g,
                           args.label_temp, args.label_sal),
            pdf_evaluation("Test", test_m, depths_g,
                           args.label_temp, args.label_sal),
            pdf_summary(val_m, test_m, args.label_temp, args.label_sal,
                        summary_history, model.n_parameters()),
        ]
        pdf_path = build_pdf(pages, out_dir,
                             title=f"Inference — epoch {epoch}")
        print(f"PDF → {pdf_path}")

    print()
    print("─" * 60)
    print(f"  Checkpoint  : {ckpt_path.name}")
    print(f"  Epoch       : {ckpt.get('epoch', '?')}")
    print(f"  Val  T RMSE : {val_m['T_rmse']:.4f} °C")
    print(f"  Val  S RMSE : {val_m['S_rmse']:.5f} PSU")
    print(f"  Test T RMSE : {test_m['T_rmse']:.4f} °C")
    print(f"  Test S RMSE : {test_m['S_rmse']:.5f} PSU")
    print("─" * 60)


if __name__ == "__main__":
    main()
