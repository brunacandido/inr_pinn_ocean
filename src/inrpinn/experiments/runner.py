"""Core training/evaluation pipeline for INR experiments.

Experiment scripts (experiment1.py, experiment2.py, …) import run_experiment
from here and only define their own parse_args() + main().

PDF and PNG plotting lives in plotting.py.
"""

from __future__ import annotations

import json
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import os as _os
_MLFLOW = False
if not _os.environ.get("NO_MLFLOW"):
    try:
        import mlflow
        import mlflow.pytorch
        _MLFLOW = True
    except ImportError:
        pass

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
    build_pdf, save_individual_pngs, save_split_map_png,
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

def _subsample_train_depths(
    result: SplitResult,
    data_fraction: float,
    seed: int,
) -> SplitResult:
    """Keep ALL surface training obs + a random fraction of non-surface training obs.

    Val and test masks are completely untouched — evaluation data is never reduced.
    """
    is_surface     = result.i_dep == 0
    deep_train_idx = np.where(result.train_mask & ~is_surface)[0]

    rng    = np.random.default_rng(seed + 9999)
    n_keep = max(1, round(data_fraction * len(deep_train_idx)))
    chosen = np.sort(rng.choice(deep_train_idx, n_keep, replace=False))

    new_train                 = result.train_mask.copy()
    new_train[deep_train_idx] = False  # drop all non-surface training obs
    new_train[chosen]         = True   # restore the randomly kept fraction

    n_surf = int((new_train & is_surface).sum())
    n_deep = int((new_train & ~is_surface).sum())
    print(f"  Training depth subsample ({data_fraction:.0%}): "
          f"{n_surf:,} surface (all kept) + {n_deep:,} deep = {n_surf + n_deep:,} total")

    return SplitResult(
        train_mask      = new_train,
        train_surf_mask = result.train_surf_mask,
        val_mask        = result.val_mask,
        test_mask       = result.test_mask,
        profile_id      = result.profile_id,
        i_lon           = result.i_lon,
        i_lat           = result.i_lat,
        i_dep           = result.i_dep,
        i_tim           = result.i_tim,
        lons            = result.lons,
        lats            = result.lats,
        depths          = result.depths,
        info            = {**result.info, "data_fraction": data_fraction},
    )


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


# ── Fast in-memory loader ─────────────────────────────────────────────────────

class _FastLoader:
    """Drop-in DataLoader replacement for in-memory numpy arrays.

    Keeps both arrays on GPU if they fit; otherwise falls back to CPU pinned
    memory with non-blocking transfer.  Eliminates multiprocessing overhead
    entirely — the main bottleneck when data already lives in RAM.
    """

    def __init__(
        self,
        coords_np: np.ndarray,
        targets_np: np.ndarray,
        batch_size: int,
        shuffle: bool,
        device: torch.device,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.device     = device
        n               = len(coords_np)
        self._n         = n

        coords_t  = torch.from_numpy(coords_np)
        targets_t = torch.from_numpy(targets_np)

        # Try GPU first (~6 floats × n × 4 bytes total)
        try:
            self._coords  = coords_t.to(device)
            self._targets = targets_t.to(device)
            self._on_gpu  = True
        except RuntimeError:
            # OOM — fall back to pinned CPU memory for fast non-blocking transfer
            self._coords  = coords_t.pin_memory()
            self._targets = targets_t.pin_memory()
            self._on_gpu  = False

    def __len__(self) -> int:
        return (self._n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        dev = self.device if self._on_gpu else torch.device("cpu")
        idx = (torch.randperm(self._n, device=dev)
               if self.shuffle else torch.arange(self._n, device=dev))
        for i in range(0, self._n, self.batch_size):
            bi = idx[i : i + self.batch_size]
            if self._on_gpu:
                yield self._coords[bi], self._targets[bi]
            else:
                yield (self._coords[bi].to(self.device, non_blocking=True),
                       self._targets[bi].to(self.device, non_blocking=True))


def _make_loader(
    *arrays: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Legacy DataLoader — kept for infer_split which uses variable-size slices."""
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
    lr      = args.lr or cfg["training"]["learning_rate"]
    use_amp = getattr(args, "amp", False) and device.type == "cuda"

    print("  Loading data into memory …", flush=True)
    train_loader = _FastLoader(
        train_coords_np, train_targets_np,
        batch_size=args.batch_size, shuffle=True, device=device,
    )
    val_loader = _FastLoader(
        val_coords_np, val_targets_np,
        batch_size=args.infer_batch, shuffle=False, device=device,
    )
    _loc = "GPU" if train_loader._on_gpu else "CPU (pinned)"
    print(f"  Training data on {_loc}  |  {len(train_loader)} batches/epoch")

    _use_fused = device.type == "cuda"
    optimiser = torch.optim.Adam(model.parameters(), lr=lr,
                                 fused=_use_fused)
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

    # Pre-allocated accumulators — in-place ops avoid new tensor creation every batch
    _tr_acc  = torch.zeros(3, device=device)  # [total, CT, SA]
    _val_acc = torch.zeros(3, device=device)

    for epoch in pbar:
        # ── train ──────────────────────────────────────────────────────────────
        model.train()
        _tr_acc.zero_()
        n_tr = 0
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
            # in-place accumulation — no new tensors, no per-batch GPU sync
            _tr_acc[0].add_(loss.detach(),       alpha=n)
            _tr_acc[1].add_(per["CT"].detach(),  alpha=n)
            _tr_acc[2].add_(per["SA"].detach(),  alpha=n)
            n_tr += n

        # single GPU→CPU transfer for all three scalars
        ep_tot, ep_T, ep_S = (_tr_acc / n_tr).tolist()
        scheduler.step()

        # ── validate (skipped on non-val epochs) ──────────────────────────────
        do_val = (epoch % val_every == 0) or (epoch == args.epochs)
        if do_val:
            model.eval()
            _val_acc.zero_()
            n_va = 0
            with torch.inference_mode():
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        _, per = model.loss(xb, yb)
                    n = xb.shape[0]
                    _val_acc[0].add_(per["CT"] + per["SA"], alpha=n)
                    _val_acc[1].add_(per["CT"],              alpha=n)
                    _val_acc[2].add_(per["SA"],              alpha=n)
                    n_va += n
            if n_va == 0:
                raise RuntimeError(
                    "Validation loader produced zero batches — val set is empty. "
                    "This should have been caught at split time; please report this as a bug."
                )
            # single transfer for all three val scalars
            v_tot, v_T, v_S = (_val_acc / n_va).tolist()
        else:
            v_tot = history["val"][-1]  if history["val"]  else float("inf")
            v_T   = history["val_T"][-1] if history["val_T"] else float("inf")
            v_S   = history["val_S"][-1] if history["val_S"] else float("inf")

        history["epoch"].append(epoch)
        history["lr"].append(scheduler.get_last_lr()[0])
        history["train"].append(ep_tot); history["train_T"].append(ep_T); history["train_S"].append(ep_S)
        history["val"].append(v_tot);   history["val_T"].append(v_T);   history["val_S"].append(v_S)

        if _MLFLOW and mlflow.active_run():
            mf_metrics = {
                "train/loss": ep_tot,
                "train/T_loss": ep_T,
                "train/S_loss": ep_S,
                "lr": scheduler.get_last_lr()[0],
            }
            if do_val:
                mf_metrics["val/loss"]   = v_tot
                mf_metrics["val/T_loss"] = v_T
                mf_metrics["val/S_loss"] = v_S
            mlflow.log_metrics(mf_metrics, step=epoch)

        if do_val and v_tot < best_val - args.min_delta:
            best_val     = v_tot
            best_epoch   = epoch
            best_state   = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        elif do_val:
            patience_ctr += 1

        if patience_ctr >= args.patience:
            stopped = epoch
            print(f"\n[epoch {epoch}] Early stop — best val={best_val:.6f} @ epoch {best_epoch}", flush=True)
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


# ── Line profiler ─────────────────────────────────────────────────────────────

def run_profiling(
    model: INR,
    train_coords_np: np.ndarray,
    train_targets_np: np.ndarray,
    val_coords_np: np.ndarray,
    val_targets_np: np.ndarray,
    args,
    cfg: dict,
    ckpt_dir: Path,
    output_dir: Path,
    device: torch.device,
    n_epochs: int,
    sync: bool = False,
) -> None:
    """Profile n_epochs of training with line_profiler and save profile.txt.

    sync=True inserts torch.cuda.synchronize() after each CUDA launch so that
    timings reflect actual GPU execution time rather than kernel launch time.
    This makes profiled code run slower than normal but gives accurate numbers.
    """
    try:
        from line_profiler import LineProfiler
    except ImportError:
        print("line_profiler not installed — run: uv add --optional dev line-profiler")
        return

    import copy, io

    prof_args = copy.copy(args)
    prof_args.epochs          = n_epochs
    prof_args.checkpoint_every = 0           # no checkpoint I/O during profiling
    prof_args.patience        = n_epochs + 1 # disable early stopping
    prof_args.val_every       = n_epochs     # validate only on the last epoch

    if sync and device.type == "cuda":
        # Patch the training loop to synchronize after every CUDA op so that
        # line_profiler sees GPU wall-clock time, not just kernel launch time.
        _orig_train = train

        def _synced_train(*a, **kw):
            import torch as _torch
            _real_step   = torch.optim.Adam.step
            _real_update = torch.amp.GradScaler.update

            def _step(self, *sa, **sk):
                r = _real_step(self, *sa, **sk)
                _torch.cuda.synchronize()
                return r

            def _update(self, *ua, **uk):
                r = _real_update(self, *ua, **uk)
                _torch.cuda.synchronize()
                return r

            torch.optim.Adam.step       = _step
            torch.amp.GradScaler.update = _update
            try:
                return _orig_train(*a, **kw)
            finally:
                torch.optim.Adam.step       = _real_step
                torch.amp.GradScaler.update = _real_update

        target = _synced_train
    else:
        target = train

    lp = LineProfiler()
    lp.add_function(train)
    for attr in ("forward", "loss"):
        fn = getattr(type(model), attr, None)
        if fn is not None:
            lp.add_function(fn)

    print(f"\nProfiling {n_epochs} epoch(s)  [sync={sync}] …")
    lp(target)(
        model,
        train_coords_np, train_targets_np,
        val_coords_np,   val_targets_np,
        prof_args, cfg, ckpt_dir, device,
    )

    stream = io.StringIO()
    lp.print_stats(stream=stream, output_unit=1e-3, stripzeros=True)
    text = stream.getvalue()
    print(text)

    out = output_dir / "profile.txt"
    out.write_text(text, encoding="utf-8")
    print(f"Profile → {out}")


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


# ── Log tee ───────────────────────────────────────────────────────────────────

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class _Tee:
    """Mirror sys.stdout to both the terminal and a log file (ANSI-stripped)."""
    def __init__(self, original, logfile):
        self._orig = original
        self._log  = logfile

    def write(self, data: str) -> int:
        self._orig.write(data)
        self._log.write(_ANSI_RE.sub("", data))
        self._log.flush()
        return len(data)

    def flush(self) -> None:
        self._orig.flush()
        self._log.flush()

    def __getattr__(self, name):
        return getattr(self._orig, name)


# ── Run-info file ─────────────────────────────────────────────────────────────

def _save_run_info(
    exp_name: str,
    val_mode: str,
    args,
    output_dir: Path,
    extra_split_kwargs: dict | None = None,
) -> Path:
    """Write run_info.txt with the full config and resume command."""
    script = {
        "uniform":          "scripts/experiment1.py",
        "contiguous":       "scripts/experiment2.py",
        "disjoint_squares": "scripts/experiment2b.py",
    }.get(val_mode, "scripts/experiment.py")

    W = 22  # key column width for alignment

    def _sec(title: str) -> list[str]:
        return ["", title, "─" * 52]

    cfg_rows = [
        ("Zarr path",        str(args.zarr_path)),
        ("Zarr group",       getattr(args, "zarr_group", "raw")),
        ("Config file",      str(args.config)),
        ("Val mode",         val_mode + (
            f" ({extra_split_kwargs['n_val_squares']} val"
            + (f" + {extra_split_kwargs['n_test_squares']} test"
               if extra_split_kwargs and "n_test_squares" in extra_split_kwargs else "")
            + " squares)"
            if extra_split_kwargs and "n_val_squares" in extra_split_kwargs else ""
        )),
        ("Train fraction",   f"{args.train_fraction}"),
        ("Val fraction",     f"{args.val_fraction}"),
        ("Seed",             str(args.seed)),
        ("Train depths frac.", str(args.train_depths_data_fraction)
                              if getattr(args, "train_depths_data_fraction", None) is not None else "—"),
        ("Weekly subsample", str(getattr(args, "weekly_subsample", False))),
        ("Epochs",           f"{args.epochs:,}"),
        ("Batch size",       f"{args.batch_size:,}"),
        ("Infer batch",      f"{args.infer_batch:,}"),
        ("Learning rate",    f"{args.lr:.2e}" if getattr(args, "lr", None) else "(from config)"),
        ("Patience",         str(args.patience)),
        ("Min delta",        str(args.min_delta)),
        ("Val every",        str(getattr(args, "val_every", 1))),
        ("Checkpoint every", str(args.checkpoint_every)),
        ("Num workers",      str(getattr(args, "num_workers", 0))),
        ("AMP",              str(getattr(args, "amp", False))),
        ("Device",           getattr(args, "device", None) or "(auto)"),
    ]

    lines: list[str] = [
        exp_name,
        "=" * len(exp_name),
        f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Output  : {output_dir}",
    ]
    lines += _sec("Configuration")
    for k, v in cfg_rows:
        lines.append(f"  {k:<{W}}{v}")

    # ── Resume command ─────────────────────────────────────────────────────────
    def _f(flag: str, val) -> str:
        return f"    --{flag} {val}"

    parts = [f"uv run {script}"]
    parts += [
        _f("zarr-path",      args.zarr_path),
        _f("zarr-group",     getattr(args, "zarr_group", "raw")),
        _f("config",         args.config),
        _f("output-dir",     args.output_dir),
        _f("train-fraction", args.train_fraction),
        _f("val-fraction",   args.val_fraction),
    ]
    if extra_split_kwargs and "n_val_squares" in extra_split_kwargs:
        parts.append(_f("n-val-squares", extra_split_kwargs["n_val_squares"]))
    if extra_split_kwargs and "n_test_squares" in extra_split_kwargs:
        parts.append(_f("n-test-squares", extra_split_kwargs["n_test_squares"]))
    parts.append(_f("seed", args.seed))
    if getattr(args, "train_depths_data_fraction", None) is not None:
        parts.append(_f("train-depths-data-fraction", args.train_depths_data_fraction))
    if getattr(args, "weekly_subsample", False):
        parts.append("    --weekly-subsample")
    parts += [
        _f("epochs",           args.epochs),
        _f("batch-size",       args.batch_size),
        _f("infer-batch",      args.infer_batch),
        _f("patience",         args.patience),
        _f("min-delta",        args.min_delta),
        _f("checkpoint-every", args.checkpoint_every),
        _f("num-workers",      getattr(args, "num_workers", 0)),
        _f("val-every",        getattr(args, "val_every", 1)),
    ]
    parts.append("    --amp" if getattr(args, "amp", False) else "    --no-amp")
    if getattr(args, "lr", None):
        parts.append(_f("lr", args.lr))
    if getattr(args, "device", None):
        parts.append(_f("device", args.device))
    parts.append(f"    --resume {output_dir}/checkpoints/epoch_XXXX.pt")

    lines += _sec("Resume command")
    lines.append(" \\\n".join(parts))
    lines.append("")

    out = output_dir / "run_info.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# ── Shared main pipeline ──────────────────────────────────────────────────────

def _check_paths(args) -> None:
    """Fail fast with a clear message if any required input path is missing."""
    errors = []

    config = Path(args.config)
    if not config.exists():
        errors.append(f"Config file not found      : {config}")
    elif config.suffix not in (".yaml", ".yml"):
        errors.append(f"Config must be a YAML file : {config}")

    zarr = Path(args.zarr_path)
    if not zarr.exists():
        errors.append(f"Zarr dataset not found     : {zarr}")
    elif not zarr.is_dir():
        errors.append(f"Zarr path must be a dir    : {zarr}")

    resume = getattr(args, "resume", None)
    if resume is not None:
        resume = Path(resume)
        if not resume.exists():
            errors.append(f"Resume checkpoint not found: {resume}")
        elif resume.suffix != ".pt":
            errors.append(f"Resume must be a .pt file  : {resume}  (got {resume.suffix!r})")
        else:
            with open(resume, "rb") as fh:
                magic = fh.read(2)
            # PyTorch saves as ZIP (b"PK") since v1.6, or legacy pickle (b"\x80")
            if magic[:1] not in (b"\x80",) and magic != b"PK":
                errors.append(
                    f"Resume file is not a PyTorch checkpoint: {resume}\n"
                    f"  First bytes: {magic!r}  — is this a PDF or log file?"
                )

    if errors:
        lines = "\n".join(f"  ✗ {e}" for e in errors)
        raise SystemExit(f"\nPath validation failed — fix these before training:\n{lines}\n")


def run_experiment(
    exp_name: str,
    val_mode: str,
    val_mode_label: str,
    args,
    extra_split_kwargs: dict | None = None,
    run_suffix: str = "",
) -> None:
    """Full experiment pipeline. Called by each experiment's main()."""
    _check_paths(args)
    cfg    = yaml.safe_load(open(args.config))
    device = get_device(args.device)
    bounds = cfg["normalisation"]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder     = f"{run_stamp}_{run_suffix}" if run_suffix else run_stamp
    output_dir = args.output_dir / folder
    plots_dir  = output_dir / "plots"
    ckpt_dir   = output_dir / "checkpoints"
    for d in (output_dir, plots_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    _save_run_info(exp_name, val_mode, args, output_dir, extra_split_kwargs)

    if _MLFLOW:
        mlflow.set_experiment(exp_name)
        _mf_run = mlflow.start_run(run_name=run_stamp)
        _mf_params = {
            "val_mode":            val_mode,
            "train_fraction":      args.train_fraction,
            "val_fraction":        args.val_fraction,
            "seed":                args.seed,
            "train_depths_data_fraction": getattr(args, "train_depths_data_fraction", None),
            "epochs":              args.epochs,
            "batch_size":          args.batch_size,
            "infer_batch":         args.infer_batch,
            "patience":            args.patience,
            "val_every":           getattr(args, "val_every", 1),
            "checkpoint_every":    args.checkpoint_every,
            "amp":                 getattr(args, "amp", False),
            "num_workers":         getattr(args, "num_workers", 0),
            "weekly_subsample":    getattr(args, "weekly_subsample", False),
        }
        if extra_split_kwargs:
            _mf_params.update(extra_split_kwargs)
        if getattr(args, "lr", None):
            _mf_params["lr"] = args.lr
        _model_cfg = cfg.get("model", {})
        _mf_params.update({f"model/{k}": v for k, v in _model_cfg.items()})
        mlflow.log_params({k: v for k, v in _mf_params.items() if v is not None})
        mlflow.set_tags({"output_dir": str(output_dir), "config": str(args.config)})
        print(f"  MLflow run: {_mf_run.info.run_id}")
    else:
        _mf_run = None

    log_path     = output_dir / "train.log"
    _log_file    = open(log_path, "w", buffering=1, encoding="utf-8")
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    sys.stdout   = _Tee(_orig_stdout, _log_file)
    sys.stderr   = _Tee(_orig_stderr, _log_file)

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
        **(extra_split_kwargs or {}),
    )
    result = splitter.split(**split_kw)
    splitter.print_summary(result)
    result = _compact_result(result)

    def _check_split(r: "SplitResult", stage: str = "") -> None:
        label = f" after {stage}" if stage else ""
        n_val  = int(r.val_mask.sum())
        n_test = int(r.test_mask.sum())
        if n_val == 0:
            raise RuntimeError(
                f"Validation set is empty{label}. The val squares contain no valid profiles. "
                "Try a larger --val-fraction, more --n-val-squares, or a different --seed."
            )
        if n_test == 0:
            raise RuntimeError(
                f"Test set is empty{label}. The test squares contain no valid profiles. "
                "Try a larger val/test split or a different --seed."
            )

    _check_split(result)

    data_fraction = getattr(args, "train_depths_data_fraction", None)
    if data_fraction is not None:
        result = _subsample_train_depths(result, data_fraction, args.seed)
        result = _compact_result(result)

    print("Saving split map …", flush=True)
    split_png = save_split_map_png(result, ds, plots_dir)
    print(f"  Split map → {split_png}")

    print("\nBuilding arrays …", flush=True)
    t0 = time.time()
    coords_np, targets_np, targets_phys = build_arrays(
        result, ds, args.var_temp, args.var_sal, bounds,
    )
    print(f"  Done ({time.time()-t0:.1f}s)  coords={coords_np.shape}")

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

    profile_epochs = getattr(args, "profile_epochs", 0)
    if profile_epochs:
        run_profiling(
            model,
            coords_np[result.train_mask], targets_np[result.train_mask],
            coords_np[result.val_mask],   targets_np[result.val_mask],
            args, cfg, ckpt_dir, output_dir, device,
            n_epochs=profile_epochs,
            sync=getattr(args, "profile_sync", False),
        )
        sys.stdout = _orig_stdout
        _log_file.close()
        return

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
        {"model_state":   best_state,
         "epoch":         best_epoch,
         "val_loss":      history["best_val_loss"],
         "cfg":           cfg,
         "split_info":    result.info,
         "split_kwargs":  {
             "mode":             val_mode,
             "train_fraction":   args.train_fraction,
             "val_fraction":     args.val_fraction,
             "seed":             args.seed,
             "weekly_subsample": getattr(args, "weekly_subsample", False),
             **(extra_split_kwargs or {}),
         },
         "history":       history,
         "training_args": {
             "exp_name":       exp_name,
             "val_mode":       val_mode,
             "val_mode_label": val_mode_label,
             "lr":             args.lr or cfg["training"]["learning_rate"],
             "batch_size":     args.batch_size,
             "epochs":         args.epochs,
             "patience":       args.patience,
             "n_params":       n_params,
             "config_name":    args.config.name,
             "zarr_name":      args.zarr_path.name,
             "train_depths_data_fraction": getattr(args, "train_depths_data_fraction", None),
         }},
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
    pdf_path = build_pdf(pages, output_dir,
                         title=f"{exp_name} — INR",
                         filename=f"{run_stamp}_report")
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

    if _MLFLOW and _mf_run is not None:
        mlflow.log_metrics({
            "final/val_T_rmse":  val_m["T_rmse"],
            "final/val_S_rmse":  val_m["S_rmse"],
            "final/test_T_rmse": test_m["T_rmse"],
            "final/test_S_rmse": test_m["S_rmse"],
            "final/best_epoch":  best_epoch,
            "final/train_min":   elapsed / 60,
            "final/n_params":    n_params,
            "final/n_train_obs": int(result.train_mask.sum()),
            "final/n_val_obs":   int(result.val_mask.sum()),
            "final/n_test_obs":  int(result.test_mask.sum()),
        })
        for path in [
            output_dir / "run_info.txt",
            output_dir / "results.json",
            output_dir / "best_model.pt",
            plots_dir  / "split_map.png",
            pdf_path,
        ]:
            if path.exists():
                mlflow.log_artifact(str(path))
        mlflow.end_run()
        print(f"MLflow run ended: {_mf_run.info.run_id}")

    sys.stdout = _orig_stdout
    sys.stderr = _orig_stderr
    _log_file.close()
    print(f"Log → {log_path}")
