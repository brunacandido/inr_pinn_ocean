#!/usr/bin/env python3
"""Experiment 4 — PINN fine-tuned from a converged INR checkpoint.

Loads a trained INR (E1 / E2 / E2B / E3), reconstructs the same data split,
warm-starts a PINN from the INR weights, and fine-tunes with the TEOS-10
equation-of-state physics loss (eq1_eos) only.

Typical usage:
    # E1 + EOS
    uv run scripts/experiment4.py \\
        --inr-checkpoint results/experiment1/20260703_140000/best_model.pt

    # Run the full E1+E2+E3 sweep:
    bash scripts/run_experiment4_eos.sh
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import trange

from inrpinn.data.splitter import GlorysProfileSplitter
from inrpinn.models.pinn import PINN
from inrpinn.models.siren import Siren
from inrpinn.experiments.runner import (
    get_device,
    load_dataset,
    build_arrays,
    infer_split,
    compute_metrics,
    _FastLoader,
    _Tee,
    _compact_result,
    _subsample_train_depths,
)
from inrpinn.experiments.plotting import (
    build_pdf,
    pdf_cover,
    pdf_split_map,
    pdf_training,
    pdf_pinn_training,
    pdf_pinn_comparison,
    pdf_evaluation,
    pdf_summary,
)

import os as _os
_MLFLOW = False
if not _os.environ.get("NO_MLFLOW"):
    try:
        import mlflow
        _MLFLOW = True
    except ImportError:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Experiment 4: PINN fine-tuned from a converged INR checkpoint. "
            "Adds EOS physics (eq1_eos) only."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Source INR checkpoint (required)
    p.add_argument("--inr-checkpoint", type=Path, required=True,
                   help="Path to best_model.pt from Experiment 1, 2, 2B, or 3.")

    # Data (can usually be inferred from the checkpoint but explicit is safer)
    p.add_argument("--zarr-path",  type=Path,
                   default=PROJECT_ROOT / "data" / "glorys_patch_34S69E.zarr")
    p.add_argument("--zarr-group", type=str, default="raw")
    p.add_argument("--config",     type=Path,
                   default=PROJECT_ROOT / "configs" / "pinn_patch_34S69E.yaml")
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "results" / "experiment4")
    p.add_argument("--var-temp",   type=str, default="thetao")
    p.add_argument("--var-sal",    type=str, default="so")
    p.add_argument("--label-temp", type=str, default="Temperature (°C)")
    p.add_argument("--label-sal",  type=str, default="Salinity (PSU)")

    # Physics
    p.add_argument("--eos-weight", type=float, default=0.1,
                   help="Weight for the TEOS-10 EOS physics loss term.")
    p.add_argument("--n-colloc",   type=int,   default=16384,
                   help="Number of collocation points sampled per batch.")

    # PINN training
    p.add_argument("--pinn-epochs",      type=int,   default=1000)
    p.add_argument("--pinn-lr",          type=float, default=1e-4)
    p.add_argument("--pinn-batch-size",  type=int,   default=8192)
    p.add_argument("--patience",         type=int,   default=100)
    p.add_argument("--min-delta",        type=float, default=1e-6)
    p.add_argument("--checkpoint-every", type=int,   default=50)
    p.add_argument("--infer-batch",      type=int,   default=32768)
    p.add_argument("--val-every",        type=int,   default=5)
    p.add_argument("--num-workers",      type=int,   default=8)
    p.add_argument("--device",           type=str,   default=None)

    if not hasattr(argparse, "_BoolOpt_defined"):
        if not hasattr(argparse, "BooleanOptionalAction"):
            class _BoolOpt(argparse.Action):
                def __init__(self, option_strings, dest, default=True, **kw):
                    opts = [o for o in option_strings if o.startswith("--")]
                    neg  = ["--no-" + o[2:] for o in opts]
                    super().__init__(opts + neg, dest, nargs=0, default=default, **kw)
                def __call__(self, parser, ns, values, opt=None):
                    del parser, values
                    setattr(ns, self.dest, not (opt or "").startswith("--no-"))
            argparse.BooleanOptionalAction = _BoolOpt  # type: ignore[attr-defined]
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pinn-resume", type=Path, default=None,
                   help="Path to a pinn_epoch_XXXX.pt checkpoint to resume PINN training from.")
    p.add_argument("--profile-epochs", type=int, default=0,
                   help="Run line_profiler for N epochs then exit (0 = disabled).")

    return p.parse_args()


# ── PINN training loop ────────────────────────────────────────────────────────

def _train_pinn(
    pinn: PINN,
    train_coords_np: np.ndarray,
    train_targets_np: np.ndarray,
    val_coords_np: np.ndarray,
    val_targets_np: np.ndarray,
    args,
    ckpt_dir: Path,
    device: torch.device,
) -> dict:
    """PINN fine-tuning loop — tracks data loss and EOS physics loss separately."""
    use_amp = getattr(args, "amp", False) and device.type == "cuda"
    lr      = args.pinn_lr

    print("  Loading data into memory …", flush=True)
    train_loader = _FastLoader(
        train_coords_np, train_targets_np,
        batch_size=args.pinn_batch_size, shuffle=True, device=device,
    )
    val_loader = _FastLoader(
        val_coords_np, val_targets_np,
        batch_size=args.infer_batch, shuffle=False, device=device,
    )
    _loc = "GPU" if train_loader._on_gpu else "CPU (pinned)"
    print(f"  Training data on {_loc}  |  {len(train_loader)} batches/epoch")

    optimiser = torch.optim.Adam(pinn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.pinn_epochs, eta_min=lr * 1e-2,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: dict[str, list] = {
        "epoch": [], "lr": [],
        "total": [], "data": [], "data_T": [], "data_S": [], "eos": [],
        "val":   [], "val_T":   [], "val_S":   [],
    }

    best_val     = float("inf")
    best_state   = {k: v.clone().cpu() for k, v in pinn.state_dict().items()}
    best_epoch   = 0
    patience_ctr = 0
    stopped      = args.pinn_epochs
    val_every    = getattr(args, "val_every", 5)
    start_epoch  = 1

    resume = getattr(args, "pinn_resume", None)
    if resume is not None:
        print(f"\nResuming PINN from {resume} …")
        rc = torch.load(resume, map_location=device, weights_only=False)
        pinn.load_state_dict(rc["model_state"])
        if "optimiser_state" in rc:
            optimiser.load_state_dict(rc["optimiser_state"])
        if "scheduler_state" in rc:
            scheduler.load_state_dict(rc["scheduler_state"])
        if rc.get("history"):
            history      = rc["history"]
        best_val     = rc.get("best_val",     best_val)
        best_epoch   = rc.get("best_epoch",   best_epoch)
        patience_ctr = rc.get("patience_ctr", patience_ctr)
        if "best_state" in rc:
            best_state = rc["best_state"]
        start_epoch  = rc["epoch"] + 1
        print(f"  Resuming from epoch {start_epoch}  |  best val so far: {best_val:.6f}")

    pbar = trange(start_epoch, args.pinn_epochs + 1, desc="PINN", unit="ep", dynamic_ncols=True)

    # Pre-allocated accumulators — in-place ops avoid new tensor creation every batch
    # train: [total, data, CT, SA, eos]   val: [total, CT, SA]
    _tr_acc  = torch.zeros(5, device=device)
    _val_acc = torch.zeros(3, device=device)
    _zero    = torch.zeros((), device=device)

    for epoch in pbar:
        # ── train step ────────────────────────────────────────────────────────
        pinn.train()
        _tr_acc.zero_()
        n_tr = 0
        colloc = PINN.sample_collocation(args.n_colloc, device)
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                total, log = pinn.loss(xb, yb, colloc)
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.step(optimiser)
            scaler.update()
            n = xb.shape[0]
            # in-place accumulation — no new tensors, no per-batch GPU sync
            _tr_acc[0].add_(log["total"],              alpha=n)
            _tr_acc[1].add_(log["data"],               alpha=n)
            _tr_acc[2].add_(log["data_CT"],            alpha=n)
            _tr_acc[3].add_(log["data_SA"],            alpha=n)
            _tr_acc[4].add_(log.get("eq1_eos", _zero), alpha=n)
            n_tr += n

        # single GPU→CPU transfer for all five scalars
        tr_vals = (_tr_acc / n_tr).tolist()
        ep_tot, ep_data, ep_T, ep_S, ep_eos = tr_vals
        scheduler.step()

        # ── validation (data loss only, same criterion as INR) ────────────────
        do_val = (epoch % val_every == 0) or (epoch == args.pinn_epochs)
        if do_val:
            pinn.eval()
            _val_acc.zero_()
            n_va = 0
            with torch.inference_mode():
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    pred = pinn(xb)
                    ct_l = F.mse_loss(pred["CT"], yb[:, 0])
                    sa_l = F.mse_loss(pred["SA"], yb[:, 1])
                    n = xb.shape[0]
                    _val_acc[0].add_(ct_l + sa_l, alpha=n)
                    _val_acc[1].add_(ct_l,         alpha=n)
                    _val_acc[2].add_(sa_l,         alpha=n)
                    n_va += n
            if n_va == 0:
                raise RuntimeError("Validation loader is empty — check the split.")
            # single transfer for all three val scalars
            v_tot, v_T, v_S = (_val_acc / n_va).tolist()
        else:
            v_tot = history["val"][-1]  if history["val"]  else float("inf")
            v_T   = history["val_T"][-1] if history["val_T"] else float("inf")
            v_S   = history["val_S"][-1] if history["val_S"] else float("inf")

        history["epoch"].append(epoch)
        history["lr"].append(scheduler.get_last_lr()[0])
        history["total"].append(ep_tot);  history["data"].append(ep_data)
        history["data_T"].append(ep_T);   history["data_S"].append(ep_S)
        history["eos"].append(ep_eos)
        history["val"].append(v_tot);  history["val_T"].append(v_T); history["val_S"].append(v_S)

        if _MLFLOW and mlflow.active_run():
            mf = {
                "pinn/total": ep_tot, "pinn/data": ep_data,
                "pinn/eos": ep_eos, "pinn/lr": scheduler.get_last_lr()[0],
            }
            if do_val:
                mf["pinn/val"] = v_tot
            mlflow.log_metrics(mf, step=epoch)

        if do_val and v_tot < best_val - args.min_delta:
            best_val     = v_tot
            best_epoch   = epoch
            best_state   = {k: v.clone().cpu() for k, v in pinn.state_dict().items()}
            patience_ctr = 0
        elif do_val:
            patience_ctr += 1

        if patience_ctr >= args.patience:
            stopped = epoch
            print(f"\n[epoch {epoch}] Early stop — best val={best_val:.6f} @ epoch {best_epoch}",
                  flush=True)
            break

        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            torch.save(
                {"epoch": epoch, "model_state": pinn.state_dict(),
                 "optimiser_state": optimiser.state_dict(),
                 "scheduler_state": scheduler.state_dict(),
                 "best_state": best_state, "best_val": best_val,
                 "best_epoch": best_epoch, "patience_ctr": patience_ctr,
                 "history": history},
                ckpt_dir / f"pinn_epoch_{epoch:04d}.pt",
            )

        pbar.set_postfix({
            "data": f"{ep_data:.5f}", "eos": f"{ep_eos:.2e}",
            "val": f"{v_tot:.5f}",   "pat": f"{patience_ctr}/{args.patience}",
        })

    history["best_epoch"]    = best_epoch
    history["best_val_loss"] = best_val
    history["stopped_epoch"] = stopped
    return history, best_state, best_epoch


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    ckpt_path = args.inr_checkpoint.resolve()
    errors = []
    if not ckpt_path.exists():
        errors.append(f"INR checkpoint not found   : {ckpt_path}")
    elif ckpt_path.suffix != ".pt":
        errors.append(f"Checkpoint must be a .pt file: {ckpt_path}  (got {ckpt_path.suffix!r})")
    else:
        with open(ckpt_path, "rb") as _fh:
            _magic = _fh.read(2)
        # PyTorch saves as ZIP (b"PK") since v1.6, or legacy pickle (b"\x80")
        if _magic[:1] not in (b"\x80",) and _magic != b"PK":
            errors.append(
                f"File is not a PyTorch checkpoint: {ckpt_path}\n"
                f"  First bytes: {_magic!r}  — did you pass a PDF or log file by mistake?"
            )
    zarr = args.zarr_path
    if not zarr.exists():
        errors.append(f"Zarr dataset not found     : {zarr}")
    config = args.config
    if not config.exists():
        errors.append(f"Config file not found      : {config}")
    if errors:
        lines = "\n".join(f"  ✗ {e}" for e in errors)
        raise SystemExit(f"\nPath validation failed:\n{lines}\n")

    print(f"Loading INR checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg     = ckpt.get("cfg") or yaml.safe_load(open(args.config))
    bounds  = cfg["normalisation"]
    device  = get_device(args.device)
    use_amp = getattr(args, "amp", False) and device.type == "cuda"

    ta           = ckpt.get("training_args", {})
    inr_exp_name = ta.get("exp_name", "INR")
    inr_val_mode_label = ta.get("val_mode_label", "—")
    inr_best_val = ckpt.get("val_loss", float("nan"))

    exp_name      = f"Experiment 4 ({inr_exp_name} + PINN EOS)"
    val_mode_label = f"{inr_val_mode_label} → PINN fine-tuning (EOS only)"

    # ── Reconstruct the exact same split ─────────────────────────────────────
    split_kw = ckpt.get("split_kwargs")
    if split_kw is None:
        print("  WARNING: checkpoint has no split_kwargs — using disjoint_squares defaults")
        split_kw = {
            "mode": "disjoint_squares", "train_fraction": 0.90,
            "val_fraction": 0.05, "seed": 42,
            "weekly_subsample": False, "n_val_squares": 3, "n_test_squares": 3,
        }

    # ── Output dirs ───────────────────────────────────────────────────────────
    run_stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_label = (
        inr_exp_name.lower()
        .replace("experiment", "e").replace(" ", "")
        .replace("(", "").replace(")", "")
    )
    run_suffix   = f"{source_label}_eos"
    output_dir   = args.output_dir / f"{run_stamp}_{run_suffix}"
    ckpt_dir     = output_dir / "checkpoints"
    plots_dir    = output_dir / "plots"
    for d in (output_dir, ckpt_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── run_info.txt ──────────────────────────────────────────────────────────
    W = 22
    def _sec(title: str) -> list[str]:
        return ["", title, "─" * 52]

    cfg_rows = [
        ("INR checkpoint",   str(ckpt_path)),
        ("INR experiment",   inr_exp_name),
        ("Zarr path",        str(args.zarr_path)),
        ("Zarr group",       args.zarr_group),
        ("Config file",      str(args.config)),
        ("Split mode",       split_kw.get("mode", "—")),
        ("Train fraction",   str(split_kw.get("train_fraction", "—"))),
        ("Val fraction",     str(split_kw.get("val_fraction", "—"))),
        ("Seed",             str(split_kw.get("seed", "—"))),
        ("EOS weight",       str(args.eos_weight)),
        ("N colloc",         str(args.n_colloc)),
        ("PINN epochs",      str(args.pinn_epochs)),
        ("Batch size",       str(args.pinn_batch_size)),
        ("Infer batch",      str(args.infer_batch)),
        ("Learning rate",    str(args.pinn_lr)),
        ("Patience",         str(args.patience)),
        ("Val every",        str(args.val_every)),
        ("Checkpoint every", str(args.checkpoint_every)),
        ("Num workers",      str(args.num_workers)),
        ("AMP",              str(getattr(args, "amp", False))),
        ("Device",           args.device or "(auto)"),
    ]

    resume_parts = [
        f"uv run scripts/experiment4.py",
        f"    --inr-checkpoint {ckpt_path}",
        f"    --zarr-path      {args.zarr_path}",
        f"    --zarr-group     {args.zarr_group}",
        f"    --config         {args.config}",
        f"    --output-dir     {args.output_dir}",
        f"    --eos-weight     {args.eos_weight}",
        f"    --n-colloc       {args.n_colloc}",
        f"    --pinn-epochs    {args.pinn_epochs}",
        f"    --pinn-lr        {args.pinn_lr}",
        f"    --pinn-batch-size {args.pinn_batch_size}",
        f"    --patience       {args.patience}",
        f"    --min-delta      {args.min_delta}",
        f"    --val-every      {args.val_every}",
        f"    --checkpoint-every {args.checkpoint_every}",
        f"    --infer-batch    {args.infer_batch}",
        f"    --num-workers    {args.num_workers}",
        ("    --amp" if getattr(args, "amp", False) else "    --no-amp"),
        f"    --pinn-resume {output_dir}/checkpoints/pinn_epoch_XXXX.pt",
    ]

    ri_lines = [
        exp_name,
        "=" * len(exp_name),
        f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Output  : {output_dir}",
    ]
    ri_lines += _sec("Configuration")
    for k, v in cfg_rows:
        ri_lines.append(f"  {k:<{W}}{v}")
    ri_lines += _sec("Rerun command")
    ri_lines.append(" \\\n".join(resume_parts))
    ri_lines.append("")

    (output_dir / "run_info.txt").write_text("\n".join(ri_lines) + "\n", encoding="utf-8")

    # ── Log tee ───────────────────────────────────────────────────────────────
    log_path     = output_dir / "train.log"
    _log_file    = open(log_path, "w", buffering=1, encoding="utf-8")
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    sys.stdout   = _Tee(_orig_stdout, _log_file)
    sys.stderr   = _Tee(_orig_stderr, _log_file)

    print("─" * 60)
    print(f"{exp_name}")
    print("─" * 60)
    print(f"  Device  : {device}")
    print(f"  INR ckpt: {ckpt_path}")
    print(f"  INR val : {inr_best_val:.6f}")
    print(f"  EOS wt  : {args.eos_weight}")
    print()

    if _MLFLOW:
        mlflow.set_experiment("Experiment 4")
        _mf_run = mlflow.start_run(run_name=run_stamp)
        mlflow.log_params({
            "inr_checkpoint": str(ckpt_path),
            "eos_weight":     args.eos_weight,
            "n_colloc":       args.n_colloc,
            "pinn_epochs":    args.pinn_epochs,
            "pinn_lr":        args.pinn_lr,
            "pinn_batch_size": args.pinn_batch_size,
            "patience":       args.patience,
            "inr_source":     inr_exp_name,
        })
    else:
        _mf_run = None

    # ── Load dataset and rebuild split ────────────────────────────────────────
    t0 = time.time()
    ds       = load_dataset(args.zarr_path, args.zarr_group)
    depths_g = ds.depth.values
    print(f"Dataset loaded ({time.time()-t0:.1f}s): {dict(ds.sizes)}")

    print(f"\nRebuilding split ({split_kw['mode']}, seed={split_kw['seed']}) …")
    splitter = GlorysProfileSplitter(ds, variables=(args.var_temp, args.var_sal))
    result   = splitter.split(**split_kw)
    splitter.print_summary(result)
    result   = _compact_result(result)

    data_frac = ta.get("train_depths_data_fraction")
    if data_frac is not None:
        result = _subsample_train_depths(result, data_frac, split_kw.get("seed", 42))
        result = _compact_result(result)
        print(f"  Applied sparsity from source experiment: {data_frac:.0%}")

    print("\nBuilding arrays …", flush=True)
    t0 = time.time()
    coords_np, targets_np, targets_phys = build_arrays(
        result, ds, args.var_temp, args.var_sal, bounds,
    )
    print(f"  Done ({time.time()-t0:.1f}s)  coords={coords_np.shape}")

    num_workers = getattr(args, "num_workers", 0)

    # ── Build PINN warm-started from INR weights ───────────────────────────────
    # Override physics_weights to EOS only; architecture from cfg
    # eq1_eos returns 0 unless the network also outputs density directly
    # (rho_norm != None). With CT+SA only, the EOS residual is always 0, so
    # we skip active_eqs entirely to avoid the expensive coords_and_grad call.
    pinn = PINN(
        siren       = Siren.from_config(cfg),
        weights     = {"eq1_eos": args.eos_weight},
        bounds      = bounds,
        masking     = cfg.get("masking", {}),
        diffusivity = cfg.get("diffusivity", {}),
        active_eqs  = [],
    ).to(device)

    # Transfer INR siren weights → PINN siren
    inr_state = ckpt["model_state"]
    pinn_siren_state = {
        k[len("siren."):]: v
        for k, v in inr_state.items()
        if k.startswith("siren.")
    }
    pinn.siren.load_state_dict(pinn_siren_state)
    n_params = pinn.n_parameters()
    print(f"\nPINN: {n_params:,} parameters  (warm-started from {ckpt_path.name})")

    # ── Evaluate INR baseline (before fine-tuning) ───────────────────────────
    print("\nEvaluating INR baseline on val/test …", flush=True)
    pinn.eval()
    T_v0, S_v0 = infer_split(pinn, coords_np, result.val_mask, device,
                              args.infer_batch, num_workers, use_amp)
    inr_val_m = compute_metrics(T_v0, S_v0, targets_phys[result.val_mask],
                                result.i_dep[result.val_mask], depths_g, bounds)
    print(f"  INR val  T RMSE: {inr_val_m['T_rmse']:.4f}°C   S RMSE: {inr_val_m['S_rmse']:.5f} PSU")

    T_t0, S_t0 = infer_split(pinn, coords_np, result.test_mask, device,
                              args.infer_batch, num_workers, use_amp)
    inr_test_m = compute_metrics(T_t0, S_t0, targets_phys[result.test_mask],
                                 result.i_dep[result.test_mask], depths_g, bounds)
    print(f"  INR test T RMSE: {inr_test_m['T_rmse']:.4f}°C   S RMSE: {inr_test_m['S_rmse']:.5f} PSU")

    # ── Optional line-profiler run ────────────────────────────────────────────
    if args.profile_epochs > 0:
        try:
            from line_profiler import LineProfiler
        except ImportError:
            raise SystemExit(
                "line_profiler not installed — run: uv add --optional dev line-profiler"
            )
        import copy, io
        prof_args = copy.copy(args)
        prof_args.pinn_epochs      = args.profile_epochs
        prof_args.checkpoint_every = 0
        prof_args.patience         = args.profile_epochs + 1
        prof_args.val_every        = args.profile_epochs

        lp = LineProfiler()
        lp.add_function(_train_pinn)
        for attr in ("forward", "loss"):
            fn = getattr(type(pinn), attr, None)
            if fn is not None:
                lp.add_function(fn)

        print(f"\nProfiling {args.profile_epochs} PINN epoch(s) …")
        lp(_train_pinn)(
            pinn,
            coords_np[result.train_mask], targets_np[result.train_mask],
            coords_np[result.val_mask],   targets_np[result.val_mask],
            prof_args, ckpt_dir, device,
        )
        buf = io.StringIO()
        lp.print_stats(stream=buf, output_unit=1e-3, stripzeros=True)
        text = buf.getvalue()
        print(text)
        prof_path = output_dir / "profile.txt"
        prof_path.write_text(text, encoding="utf-8")
        print(f"Profile → {prof_path}")
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        _log_file.close()
        return

    # ── PINN fine-tuning ──────────────────────────────────────────────────────
    print("\nStarting PINN fine-tuning …")
    t0 = time.time()
    history, best_state, best_epoch = _train_pinn(
        pinn,
        coords_np[result.train_mask], targets_np[result.train_mask],
        coords_np[result.val_mask],   targets_np[result.val_mask],
        args, ckpt_dir, device,
    )
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min  |  best epoch {best_epoch}  |  "
          f"best val {history['best_val_loss']:.6f}")

    # ── Save PINN checkpoint ──────────────────────────────────────────────────
    torch.save(
        {"model_state":    best_state,
         "epoch":          best_epoch,
         "val_loss":       history["best_val_loss"],
         "cfg":            cfg,
         "split_kwargs":   split_kw,
         "history":        history,
         "inr_val_metrics":  {k: v for k, v in inr_val_m.items()
                              if not isinstance(v, np.ndarray)},
         "inr_test_metrics": {k: v for k, v in inr_test_m.items()
                              if not isinstance(v, np.ndarray)},
         "training_args": {
             "exp_name":        exp_name,
             "val_mode_label":  val_mode_label,
             "lr":              args.pinn_lr,
             "batch_size":      args.pinn_batch_size,
             "epochs":          args.pinn_epochs,
             "patience":        args.patience,
             "n_params":        n_params,
             "eos_weight":      args.eos_weight,
             "n_colloc":        args.n_colloc,
             "inr_checkpoint":  str(ckpt_path),
             "inr_exp_name":    inr_exp_name,
         }},
        output_dir / "best_pinn.pt",
    )

    pinn.load_state_dict(best_state)

    # ── Evaluate PINN ─────────────────────────────────────────────────────────
    print("\nEvaluating PINN on val/test …", flush=True)
    T_vp, S_vp = infer_split(pinn, coords_np, result.val_mask, device,
                              args.infer_batch, num_workers, use_amp)
    val_m = compute_metrics(T_vp, S_vp, targets_phys[result.val_mask],
                            result.i_dep[result.val_mask], depths_g, bounds)
    print(f"  PINN val  T RMSE: {val_m['T_rmse']:.4f}°C   S RMSE: {val_m['S_rmse']:.5f} PSU")

    T_tp, S_tp = infer_split(pinn, coords_np, result.test_mask, device,
                              args.infer_batch, num_workers, use_amp)
    test_m = compute_metrics(T_tp, S_tp, targets_phys[result.test_mask],
                             result.i_dep[result.test_mask], depths_g, bounds)
    print(f"  PINN test T RMSE: {test_m['T_rmse']:.4f}°C   S RMSE: {test_m['S_rmse']:.5f} PSU")

    # ── Save results JSON ─────────────────────────────────────────────────────
    def _clean(d: dict) -> dict:
        return {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}

    results = {
        "experiment":    exp_name,
        "run_time":      datetime.now().isoformat(),
        "inr_checkpoint": str(ckpt_path),
        "eos_weight":    args.eos_weight,
        "n_colloc":      args.n_colloc,
        "training": {
            "best_epoch": best_epoch,
            "best_val_loss": history["best_val_loss"],
            "stopped_epoch": history["stopped_epoch"],
        },
        "inr_metrics": {
            "validation": _clean(inr_val_m), "test": _clean(inr_test_m),
        },
        "pinn_metrics": {
            "validation": _clean(val_m), "test": _clean(test_m),
        },
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults → {results_path}")

    # ── INR training history (from the source INR checkpoint) ─────────────────
    inr_history = ckpt.get("history")

    # ── Build PDF report ──────────────────────────────────────────────────────
    print("Building PDF …", flush=True)

    cover_args = types.SimpleNamespace(
        lr          = args.pinn_lr,
        batch_size  = args.pinn_batch_size,
        epochs      = args.pinn_epochs,
        patience    = args.patience,
        seed        = split_kw.get("seed", 42),
        config      = types.SimpleNamespace(name=args.config.name),
        zarr_path   = types.SimpleNamespace(name=args.zarr_path.name),
    )

    # Cover history: use PINN history with data loss as the "train/val" loss
    cover_history = {
        **history,
        "train":   history["data"],
        "train_T": history["data_T"],
        "train_S": history["data_S"],
    }

    pages = [
        pdf_cover(
            exp_name        = exp_name,
            val_mode_label  = val_mode_label,
            args            = cover_args,
            cfg             = cfg,
            n_params        = n_params,
            result          = result,
            history         = cover_history,
            val_metrics     = val_m,
            test_metrics    = test_m,
        ),
        pdf_split_map(result, ds),
    ]

    # INR training history (source experiment)
    if inr_history is not None:
        pages.append(pdf_training(inr_history))

    pages += [
        pdf_pinn_training(history, inr_best_val=inr_best_val),
        pdf_pinn_comparison(
            inr_val_m, inr_test_m, val_m, test_m,
            args.label_temp, args.label_sal, depths_g,
        ),
        pdf_evaluation("PINN Validation", val_m, depths_g,
                       args.label_temp, args.label_sal),
        pdf_evaluation("PINN Test",       test_m, depths_g,
                       args.label_temp, args.label_sal),
        pdf_summary(val_m, test_m, args.label_temp, args.label_sal,
                    cover_history, n_params),
    ]

    pdf_path = build_pdf(
        pages, output_dir,
        title=f"{exp_name}",
        filename=f"{run_stamp}_report",
    )
    print(f"PDF → {pdf_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print("─" * 60)
    print(f"{exp_name} complete")
    print("─" * 60)
    print(f"  INR  val  T RMSE : {inr_val_m['T_rmse']:.4f} °C")
    print(f"  INR  val  S RMSE : {inr_val_m['S_rmse']:.5f} PSU")
    print(f"  PINN val  T RMSE : {val_m['T_rmse']:.4f} °C  "
          f"  ({val_m['T_rmse'] - inr_val_m['T_rmse']:+.4f})")
    print(f"  PINN val  S RMSE : {val_m['S_rmse']:.5f} PSU"
          f"  ({val_m['S_rmse'] - inr_val_m['S_rmse']:+.5f})")
    print(f"  INR  test T RMSE : {inr_test_m['T_rmse']:.4f} °C")
    print(f"  INR  test S RMSE : {inr_test_m['S_rmse']:.5f} PSU")
    print(f"  PINN test T RMSE : {test_m['T_rmse']:.4f} °C  "
          f"  ({test_m['T_rmse'] - inr_test_m['T_rmse']:+.4f})")
    print(f"  PINN test S RMSE : {test_m['S_rmse']:.5f} PSU"
          f"  ({test_m['S_rmse'] - inr_test_m['S_rmse']:+.5f})")
    print(f"  Best epoch       : {best_epoch}")
    print(f"  Training         : {elapsed/60:.1f} min")
    print("─" * 60)

    if _MLFLOW and _mf_run is not None:
        mlflow.log_metrics({
            "final/inr_val_T_rmse":   inr_val_m["T_rmse"],
            "final/inr_val_S_rmse":   inr_val_m["S_rmse"],
            "final/pinn_val_T_rmse":  val_m["T_rmse"],
            "final/pinn_val_S_rmse":  val_m["S_rmse"],
            "final/inr_test_T_rmse":  inr_test_m["T_rmse"],
            "final/inr_test_S_rmse":  inr_test_m["S_rmse"],
            "final/pinn_test_T_rmse": test_m["T_rmse"],
            "final/pinn_test_S_rmse": test_m["S_rmse"],
        })
        mlflow.log_artifact(str(output_dir / "best_pinn.pt"))
        mlflow.log_artifact(str(results_path))
        mlflow.end_run()

    sys.stdout = _orig_stdout
    sys.stderr = _orig_stderr
    _log_file.close()


if __name__ == "__main__":
    main()
