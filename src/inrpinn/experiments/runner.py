"""Core training/evaluation pipeline for INR experiments.

Experiment scripts (experiment1.py, experiment2.py, …) import run_experiment
from here and only define their own parse_args() + main().

PDF and PNG plotting lives in plotting.py.
"""

from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import xarray as xr
import yaml
from tqdm.auto import trange

from inrpinn.data.dataset import denorm_minmax, norm_minmax
from inrpinn.data.splitter import GlorysProfileSplitter, SplitResult
from inrpinn.models.inr import INR
from inrpinn.experiments.plotting import (
    build_pdf, save_individual_pngs,
    pdf_cover, pdf_spatial_fields, pdf_temporal_profiles,
    pdf_split_map, pdf_training, pdf_evaluation, pdf_summary,
)


# ── Device ────────────────────────────────────────────────────────────────────

def get_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Metric helpers ────────────────────────────────────────────────────────────

def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


# ── Data ──────────────────────────────────────────────────────────────────────

def load_dataset(zarr_path: Path, group: str) -> xr.Dataset:
    return xr.open_zarr(zarr_path / group, consolidated=False)


def build_arrays(
    result: SplitResult,
    ds: xr.Dataset,
    var_temp: str,
    var_sal: str,
    bounds: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (coords_norm, targets_norm, targets_phys) aligned with result indices."""
    i_lon, i_lat, i_dep, i_tim = result.i_lon, result.i_lat, result.i_dep, result.i_tim

    T_4d = ds[var_temp].transpose("longitude", "latitude", "depth", "time").values
    S_4d = ds[var_sal].transpose("longitude", "latitude", "depth", "time").values

    T_flat = T_4d[i_lon, i_lat, i_dep, i_tim].astype(np.float32)
    S_flat = S_4d[i_lon, i_lat, i_dep, i_tim].astype(np.float32)
    del T_4d, S_4d

    targets_phys = np.stack([T_flat, S_flat], axis=1)

    coords_np = np.stack([
        norm_minmax(result.lons,              *bounds["lon"]),
        norm_minmax(result.lats,              *bounds["lat"]),
        norm_minmax(result.depths,            *bounds["depth"]),
        norm_minmax(i_tim.astype(np.float32), *bounds["time"]),
    ], axis=1).astype(np.float32)

    targets_np = np.stack([
        norm_minmax(T_flat, *bounds["T"]),
        norm_minmax(S_flat, *bounds["S"]),
    ], axis=1).astype(np.float32)

    return coords_np, targets_np, targets_phys


# ── Depth subsampling ─────────────────────────────────────────────────────────

def _subsample_depths(
    result: SplitResult,
    coords_np: np.ndarray,
    targets_np: np.ndarray,
    targets_phys: np.ndarray,
    depth_fraction: float,
    seed: int,
) -> tuple[SplitResult, np.ndarray, np.ndarray, np.ndarray]:
    """Keep all surface (depth index 0) observations and a random fraction of deeper ones."""
    is_surface = result.i_dep == 0
    deep_idx   = np.where(~is_surface)[0]

    rng    = np.random.default_rng(seed + 9999)
    n_keep = max(1, round(depth_fraction * len(deep_idx)))
    chosen = rng.choice(deep_idx, n_keep, replace=False)

    keep          = is_surface.copy()
    keep[chosen]  = True

    new_result = SplitResult(
        train_mask      = result.train_mask[keep],
        train_surf_mask = result.train_surf_mask[keep],
        val_mask        = result.val_mask[keep],
        test_mask       = result.test_mask[keep],
        profile_id      = result.profile_id[keep],
        i_lon           = result.i_lon[keep],
        i_lat           = result.i_lat[keep],
        i_dep           = result.i_dep[keep],
        i_tim           = result.i_tim[keep],
        lons            = result.lons[keep],
        lats            = result.lats[keep],
        depths          = result.depths[keep],
        info            = {**result.info, "depth_fraction": depth_fraction},
    )
    return (
        new_result,
        coords_np[keep],
        targets_np[keep],
        targets_phys[keep],
    )


# ── Compact result (drop unassigned observations) ─────────────────────────────

def _compact_result(result: SplitResult) -> SplitResult:
    """Return a SplitResult keeping only observations assigned to at least one split.

    When data_fraction < 1, most observations have all masks False.
    Compacting before build_arrays reduces flat-array memory by ~1/data_fraction.
    """
    used = result.train_mask | result.val_mask | result.test_mask | result.train_surf_mask
    if used.all():
        return result
    return SplitResult(
        train_mask      = result.train_mask[used],
        train_surf_mask = result.train_surf_mask[used],
        val_mask        = result.val_mask[used],
        test_mask       = result.test_mask[used],
        profile_id      = result.profile_id[used],
        i_lon           = result.i_lon[used],
        i_lat           = result.i_lat[used],
        i_dep           = result.i_dep[used],
        i_tim           = result.i_tim[used],
        lons            = result.lons[used],
        lats            = result.lats[used],
        depths          = result.depths[used],
        info            = result.info,
    )


# ── DataLoader factory ────────────────────────────────────────────────────────

def _make_loader(
    *arrays: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    tensors = [torch.from_numpy(a) for a in arrays]
    kw: dict = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        kw["prefetch_factor"]    = 2
        kw["persistent_workers"] = True
    return DataLoader(TensorDataset(*tensors), **kw)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    model: INR,
    train_coords_np: np.ndarray,
    train_targets_np: np.ndarray,
    val_coords_np: np.ndarray,
    val_targets_np: np.ndarray,
    args,
    cfg: dict,
    ckpt_dir: Path,
    device: torch.device,
    resume_from: Path | None = None,
) -> tuple[dict, dict, int]:
    lr          = args.lr or cfg["training"]["learning_rate"]
    use_amp     = getattr(args, "amp", False) and device.type == "cuda"
    num_workers = getattr(args, "num_workers", 0)
    pin_memory  = device.type == "cuda"

    train_loader = _make_loader(
        train_coords_np, train_targets_np,
        batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = _make_loader(
        val_coords_np, val_targets_np,
        batch_size=args.infer_batch, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=lr * 1e-2,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: dict[str, list] = {
        "epoch": [], "lr": [],
        "train": [], "train_T": [], "train_S": [],
        "val":   [], "val_T":   [], "val_S":   [],
    }

    best_val     = float("inf")
    best_state   = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    best_epoch   = 0
    patience_ctr = 0
    stopped      = args.epochs
    val_every    = getattr(args, "val_every", 1)
    start_epoch  = 1

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if resume_from is not None:
        print(f"\nResuming from {resume_from} …")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "optimiser_state" in ckpt:
            optimiser.load_state_dict(ckpt["optimiser_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if "history" in ckpt and ckpt["history"]:
            history      = ckpt["history"]
        best_val     = ckpt.get("best_val",     best_val)
        best_epoch   = ckpt.get("best_epoch",   best_epoch)
        patience_ctr = ckpt.get("patience_ctr", patience_ctr)
        if "best_state" in ckpt:
            best_state = ckpt["best_state"]
        start_epoch  = ckpt["epoch"] + 1
        print(f"  Resuming from epoch {start_epoch}  |  best val so far: {best_val:.6f}")

    pbar = trange(start_epoch, args.epochs + 1, desc="Training", unit="ep", dynamic_ncols=True)

    for epoch in pbar:
        # ── train ──────────────────────────────────────────────────────────────
        model.train()
        ep_tot = ep_T = ep_S = n_tr = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss, per = model.loss(xb, yb)
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            n = xb.shape[0]
            ep_tot += loss.detach().item() * n
            ep_T   += per["CT"].detach().item() * n
            ep_S   += per["SA"].detach().item() * n
            n_tr   += n

        ep_tot /= n_tr;  ep_T /= n_tr;  ep_S /= n_tr
        scheduler.step()

        # ── validate (skipped on non-val epochs) ──────────────────────────────
        do_val = (epoch % val_every == 0) or (epoch == args.epochs)
        if do_val:
            model.eval()
            v_tot = v_T = v_S = n_va = 0
            with torch.inference_mode():
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        _, per = model.loss(xb, yb)
                    n = xb.shape[0]
                    v_tot += (per["CT"].item() + per["SA"].item()) * n
                    v_T   += per["CT"].item() * n
                    v_S   += per["SA"].item() * n
                    n_va  += n
            v_tot /= n_va;  v_T /= n_va;  v_S /= n_va
        else:
            v_tot = history["val"][-1]  if history["val"]  else float("inf")
            v_T   = history["val_T"][-1] if history["val_T"] else float("inf")
            v_S   = history["val_S"][-1] if history["val_S"] else float("inf")

        history["epoch"].append(epoch)
        history["lr"].append(scheduler.get_last_lr()[0])
        history["train"].append(ep_tot); history["train_T"].append(ep_T); history["train_S"].append(ep_S)
        history["val"].append(v_tot);   history["val_T"].append(v_T);   history["val_S"].append(v_S)

        if do_val and v_tot < best_val - args.min_delta:
            best_val     = v_tot
            best_epoch   = epoch
            best_state   = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        elif do_val:
            patience_ctr += 1

        if patience_ctr >= args.patience:
            stopped = epoch
            pbar.write(f"[epoch {epoch}] Early stop — best val={best_val:.6f} @ epoch {best_epoch}")
            break

        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            torch.save(
                {"epoch":           epoch,
                 "model_state":     model.state_dict(),
                 "optimiser_state": optimiser.state_dict(),
                 "scheduler_state": scheduler.state_dict(),
                 "best_state":      best_state,
                 "best_val":        best_val,
                 "best_epoch":      best_epoch,
                 "patience_ctr":    patience_ctr,
                 "val_loss":        v_tot,
                 "cfg":             cfg,
                 "history":         history},
                ckpt_dir / f"epoch_{epoch:04d}.pt",
            )

        pbar.set_postfix({
            "tr": f"{ep_tot:.5f}", "va": f"{v_tot:.5f}",
            "best": f"{best_val:.5f}", "pat": f"{patience_ctr}/{args.patience}",
        })

    history["best_epoch"]    = best_epoch
    history["best_val_loss"] = best_val
    history["stopped_epoch"] = stopped
    return history, best_state, best_epoch


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.inference_mode()
def infer_split(
    model: INR,
    coords_np: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    infer_batch: int,
    num_workers: int = 0,
    use_amp: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    loader = _make_loader(
        coords_np[mask],
        batch_size=infer_batch, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    n  = int(mask.sum())
    T_ = np.empty(n, dtype=np.float32)
    S_ = np.empty(n, dtype=np.float32)
    model.eval()
    pos = 0
    for (xb,) in loader:
        xb  = xb.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(xb)
        end = pos + xb.shape[0]
        T_[pos:end] = out["CT"].float().cpu().numpy()
        S_[pos:end] = out["SA"].float().cpu().numpy()
        pos = end
    return T_, S_


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    T_pred_norm: np.ndarray, S_pred_norm: np.ndarray,
    targets_phys_sub: np.ndarray,
    i_dep_sub: np.ndarray, depths_g: np.ndarray,
    bounds: dict,
) -> dict:
    T_pred = denorm_minmax(T_pred_norm, *bounds["T"])
    S_pred = denorm_minmax(S_pred_norm, *bounds["S"])
    T_true = targets_phys_sub[:, 0]
    S_true = targets_phys_sub[:, 1]

    n_dep = len(depths_g)
    T_rd  = np.full(n_dep, np.nan)
    S_rd  = np.full(n_dep, np.nan)
    T_md  = np.full(n_dep, np.nan)
    S_md  = np.full(n_dep, np.nan)

    for k in range(n_dep):
        sel = i_dep_sub == k
        if sel.sum() < 2:
            continue
        T_rd[k] = _rmse(T_pred[sel], T_true[sel])
        S_rd[k] = _rmse(S_pred[sel], S_true[sel])
        T_md[k] = _mae(T_pred[sel],  T_true[sel])
        S_md[k] = _mae(S_pred[sel],  S_true[sel])

    return {
        "T_rmse": _rmse(T_pred, T_true), "S_rmse": _rmse(S_pred, S_true),
        "T_mae":  _mae(T_pred,  T_true), "S_mae":  _mae(S_pred,  S_true),
        "T_rmse_depth": T_rd.tolist(),   "S_rmse_depth": S_rd.tolist(),
        "T_mae_depth":  T_md.tolist(),   "S_mae_depth":  S_md.tolist(),
        "T_pred": T_pred, "S_pred": S_pred,
        "T_true": T_true, "S_true": S_true,
        "n_obs":  int(len(T_pred)),
    }


# ── Persist results ───────────────────────────────────────────────────────────

def save_results(
    exp_name: str,
    val_mode: str,
    args,
    cfg: dict,
    split_info: dict,
    history: dict,
    val_metrics: dict,
    test_metrics: dict,
    output_dir: Path,
) -> Path:
    def _clean(d: dict) -> dict:
        return {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}

    results = {
        "experiment":  exp_name,
        "run_time":    datetime.now().isoformat(),
        "config_file": str(args.config),
        "zarr_path":   str(args.zarr_path),
        "seed":        args.seed,
        "split": {
            "mode":           val_mode,
            "train_fraction": args.train_fraction,
            "val_fraction":   args.val_fraction,
            **{k: v for k, v in split_info.items() if not isinstance(v, np.ndarray)},
        },
        "model": cfg.get("model", {}),
        "training": {
            "lr":            args.lr or cfg["training"]["learning_rate"],
            "epochs":        args.epochs,
            "batch_size":    args.batch_size,
            "patience":      args.patience,
            "min_delta":     args.min_delta,
            "best_epoch":    history["best_epoch"],
            "best_val_loss": history["best_val_loss"],
            "stopped_epoch": history["stopped_epoch"],
        },
        "metrics": {
            "validation": _clean(val_metrics),
            "test":       _clean(test_metrics),
        },
        "history": {
            k: v for k, v in history.items()
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float))
        },
    }

    out = output_dir / "results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return out


# ── Shared main pipeline ──────────────────────────────────────────────────────

def run_experiment(
    exp_name: str,
    val_mode: str,
    val_mode_label: str,
    args,
    extra_split_kwargs: dict | None = None,
) -> None:
    """Full experiment pipeline. Called by each experiment's main()."""
    cfg    = yaml.safe_load(open(args.config))
    device = get_device(args.device)
    bounds = cfg["normalisation"]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_stamp
    plots_dir  = output_dir / "plots"
    ckpt_dir   = output_dir / "checkpoints"
    for d in (output_dir, plots_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("─" * 60)
    print(f"{exp_name} — INR  |  {val_mode} split")
    print("─" * 60)
    print(f"  Device  : {device}")
    print(f"  Config  : {args.config.name}")
    print(f"  Zarr    : {args.zarr_path}")
    print(f"  Output  : {output_dir}")
    print()

    t0 = time.time()
    ds       = load_dataset(args.zarr_path, args.zarr_group)
    depths_g = ds.depth.values
    print(f"Dataset loaded ({time.time()-t0:.1f}s): {dict(ds.sizes)}")

    print(f"\nBuilding profile split ({val_mode}) …")
    splitter = GlorysProfileSplitter(ds, variables=(args.var_temp, args.var_sal))
    split_kw = dict(
        mode=val_mode,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
        weekly_subsample=getattr(args, "weekly_subsample", False),
        data_fraction=getattr(args, "data_fraction", None),
        **(extra_split_kwargs or {}),
    )
    result = splitter.split(**split_kw)
    splitter.print_summary(result)
    result = _compact_result(result)

    print("\nBuilding arrays …", flush=True)
    t0 = time.time()
    coords_np, targets_np, targets_phys = build_arrays(
        result, ds, args.var_temp, args.var_sal, bounds,
    )
    print(f"  Done ({time.time()-t0:.1f}s)  coords={coords_np.shape}")

    depth_fraction = getattr(args, "depth_fraction", None)
    if depth_fraction is not None:
        result, coords_np, targets_np, targets_phys = _subsample_depths(
            result, coords_np, targets_np, targets_phys, depth_fraction, args.seed,
        )
        n_surf = int((result.i_dep == 0).sum())
        n_deep = int((result.i_dep > 0).sum())
        print(f"  Depth subsample ({depth_fraction:.0%}): "
              f"{n_surf:,} surface + {n_deep:,} deep = {len(coords_np):,} total obs")

    num_workers = getattr(args, "num_workers", 0)
    use_amp     = getattr(args, "amp", False) and device.type == "cuda"

    model    = INR.from_config(cfg).to(device)
    n_params = model.n_parameters()

    if getattr(args, "compile", False) and device.type == "cuda":
        model = torch.compile(model)
        print(f"\nModel (compiled): {model}  |  params: {n_params:,}")
    else:
        print(f"\nModel: {model}  |  params: {n_params:,}")

    print(f"  AMP={use_amp}  workers={num_workers}")

    print("\nStarting training …")
    t0 = time.time()
    history, best_state, best_epoch = train(
        model,
        coords_np[result.train_mask], targets_np[result.train_mask],
        coords_np[result.val_mask],   targets_np[result.val_mask],
        args, cfg, ckpt_dir, device,
        resume_from=getattr(args, "resume", None),
    )
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min  |  best epoch {best_epoch}  |  "
          f"best val {history['best_val_loss']:.6f}")

    torch.save(
        {"model_state": best_state, "epoch": best_epoch,
         "val_loss": history["best_val_loss"], "cfg": cfg,
         "split_info": result.info},
        output_dir / "best_model.pt",
    )

    model.load_state_dict(best_state)

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

    results_path = save_results(exp_name, val_mode, args, cfg,
                                result.info, history, val_m, test_m, output_dir)
    print(f"\nResults → {results_path}")

    print("Saving individual plots …", flush=True)
    save_individual_pngs(ds, result, depths_g,
                         args.var_temp, args.var_sal,
                         args.label_temp, args.label_sal,
                         history, val_m, test_m, plots_dir)

    print("Building PDF …", flush=True)
    pages = [
        pdf_cover(exp_name, val_mode_label, args, cfg, n_params,
                  result, history, val_m, test_m),
        pdf_spatial_fields(ds, args.var_temp, args.var_sal,
                           args.label_temp, args.label_sal),
        pdf_temporal_profiles(ds, result, depths_g,
                              args.var_temp, args.var_sal,
                              args.label_temp, args.label_sal),
        pdf_split_map(result, ds),
        pdf_training(history),
        pdf_evaluation("Validation", val_m, depths_g, args.label_temp, args.label_sal),
        pdf_evaluation("Test",       test_m, depths_g, args.label_temp, args.label_sal),
        pdf_summary(val_m, test_m, args.label_temp, args.label_sal, history, n_params),
    ]
    pdf_path = build_pdf(pages, output_dir, title=f"{exp_name} — INR")
    print(f"PDF → {pdf_path}")

    print()
    print("─" * 60)
    print(f"{exp_name} complete")
    print("─" * 60)
    print(f"  Val  T RMSE : {val_m['T_rmse']:.4f} °C")
    print(f"  Val  S RMSE : {val_m['S_rmse']:.5f} PSU")
    print(f"  Test T RMSE : {test_m['T_rmse']:.4f} °C")
    print(f"  Test S RMSE : {test_m['S_rmse']:.5f} PSU")
    print(f"  Best epoch  : {best_epoch}")
    print(f"  Training    : {elapsed/60:.1f} min")
    print("─" * 60)
