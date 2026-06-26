#!/usr/bin/env python3
"""Experiment 1 — INR baseline: dense training + uniform validation split.

Split   : GlorysProfileSplitter  mode="uniform"
          Profiles (entire water columns) are never split across sets.
Model   : INR wrapping a SIREN backbone — data MSE loss only, no physics.
Stopping: early stopping on total validation loss (patience + min-delta).

Outputs → --output-dir/
  checkpoints/epoch_{N:04d}.pt   periodic weight snapshots
  best_model.pt                   weights at lowest validation loss seen
  plots/<name>.png                individual figure files
  results.json                    all numeric metrics and run configuration
  report.pdf                      A4 multi-page PDF report
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from tqdm.auto import trange

from inrpinn.data.dataset import denorm_minmax, norm_minmax
from inrpinn.data.splitter import GlorysProfileSplitter, SplitResult
from inrpinn.models.inr import INR

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# A4 page dimensions (inches)
A4_P = (8.27, 11.69)   # portrait
A4_L = (11.69, 8.27)   # landscape

# Colour palette
CMAP_T = "RdYlBu_r"
CMAP_S = "viridis"
COL_T  = "#c0392b"   # temperature — deep red
COL_S  = "#2980b9"   # salinity    — deep blue
COL_TR = "#2c7bb6"   # training loss
COL_VA = "#d7191c"   # validation loss

plt.rcParams.update({
    "font.size":        9,
    "axes.titlesize":  10,
    "axes.labelsize":   9,
    "legend.fontsize":  8,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "axes.grid":       True,
    "grid.alpha":      0.35,
})


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experiment 1: INR with dense training and uniform validation split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--zarr-path", type=Path,
                   default=PROJECT_ROOT / "data" / "glorys_patch_34S69E.zarr")
    p.add_argument("--zarr-group", type=str, default="raw")
    p.add_argument("--config", type=Path,
                   default=PROJECT_ROOT / "configs" / "pinn_patch_34S69E.yaml")
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "results" / "experiment1")
    p.add_argument("--var-temp",   type=str, default="thetao")
    p.add_argument("--var-sal",    type=str, default="so")
    p.add_argument("--label-temp", type=str, default="Temperature (°C)")
    p.add_argument("--label-sal",  type=str, default="Salinity (PSU)")

    # Split
    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--val-fraction",   type=float, default=0.15)
    p.add_argument("--seed",           type=int,   default=42)

    # Training
    p.add_argument("--epochs",           type=int,   default=2000)
    p.add_argument("--batch-size",       type=int,   default=8192)
    p.add_argument("--lr",               type=float, default=None)
    p.add_argument("--patience",         type=int,   default=100)
    p.add_argument("--min-delta",        type=float, default=1e-6)
    p.add_argument("--checkpoint-every", type=int,   default=50)
    p.add_argument("--infer-batch",      type=int,   default=32768)
    p.add_argument("--device",           type=str,   default=None)

    return p.parse_args()


# ── Utilities ────────────────────────────────────────────────────────────────

def get_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _save_fig(fig: plt.Figure, plots_dir: Path, name: str) -> Path:
    p = plots_dir / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# ── Data ─────────────────────────────────────────────────────────────────────

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


# ── Low-level draw helpers ────────────────────────────────────────────────────
# Each function draws onto a supplied Axes object and returns nothing.

def _draw_surface_field(ax, lons, lats, data, label, cmap, title):
    im = ax.pcolormesh(lons, lats, data, cmap=cmap, shading="auto")
    plt.colorbar(im, ax=ax, label=label, pad=0.02, fraction=0.046)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(title)


def _draw_timeseries(ax_T, ax_S, times, sst, sss, label_T, label_S):
    ax_T.plot(times, sst, color=COL_T, lw=1.4)
    ax_T.set_ylabel(label_T)
    ax_T.set_title("Domain-mean sea surface fields (annual)")
    ax_S.plot(times, sss, color=COL_S, lw=1.4)
    ax_S.set_ylabel(label_S)
    for ax in (ax_T, ax_S):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax_T.tick_params(labelbottom=False)


def _draw_profile(ax, prof, depths, label, color):
    ax.plot(prof, depths, "o-", color=color, ms=2.5, lw=1.4)
    ax.invert_yaxis()
    ax.set_xlabel(label)
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"Mean {label} profile")


def _draw_depth_counts(ax, i_dep, depths):
    counts = np.bincount(i_dep, minlength=len(depths))
    ax.barh(depths, counts / 1e3, color="steelblue", alpha=0.75)
    ax.invert_yaxis()
    ax.set_xlabel("Obs (×10³)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Valid obs per depth")


def _draw_split_frac(ax, lons, lats, frac_map, title, cmap):
    im = ax.pcolormesh(lons, lats, frac_map.T, cmap=cmap,
                       vmin=0, vmax=1, shading="auto")
    plt.colorbar(im, ax=ax, label="Fraction", pad=0.02, fraction=0.046)
    ax.set_title(title)
    ax.set_xlabel("Lon (°E)")
    ax.set_ylabel("Lat (°N)")


def _draw_loss_curve(ax, epochs, train_vals, val_vals, title,
                     best_ep, color_tr=COL_TR, color_va=COL_VA):
    ax.semilogy(epochs, train_vals, lw=1.4, color=color_tr, label="Train")
    ax.semilogy(epochs, val_vals,   lw=1.4, color=color_va, label="Val", ls="--")
    ax.axvline(best_ep, color="#555", ls=":", lw=1, label=f"Best ({best_ep})")
    ax.set_ylabel("MSE (log)")
    ax.set_title(title)
    ax.legend(loc="upper right")


def _draw_lr_curve(ax, epochs, lr_vals):
    ax.semilogy(epochs, lr_vals, color="#27ae60", lw=1.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR (log)")
    ax.set_title("Learning rate schedule")


def _draw_scatter(ax, pred, true, label, unit, rmse, mae, color, rng):
    n = min(50_000, len(pred))
    idx = rng.choice(len(pred), n, replace=False)
    ax.scatter(true[idx], pred[idx], s=2, alpha=0.25, color=color,
               edgecolors="none", rasterized=True)
    lo = min(true.min(), pred.min())
    hi = max(true.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="1 : 1")
    ax.set_xlabel(f"GLORYS {label}")
    ax.set_ylabel(f"INR {label}")
    ax.set_title(
        f"{label}\n"
        f"RMSE = {rmse:.4f} {unit}   MAE = {mae:.4f} {unit}"
    )
    ax.legend(fontsize=7)


def _draw_rmse_depth(ax, rmse_d, depths, label, unit, color):
    valid = ~np.isnan(rmse_d)
    ax.plot(rmse_d[valid], depths[valid], "o-", color=color, ms=3, lw=1.4)
    ax.invert_yaxis()
    ax.set_xlabel(f"RMSE ({unit})")
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"{label} RMSE vs depth\nOverall = {np.nanmean(rmse_d):.4f} {unit}")


# ── PDF page builders ─────────────────────────────────────────────────────────
# Each returns a Figure that maps to one A4 page in the PDF.

def pdf_cover(
    args: argparse.Namespace,
    cfg: dict,
    n_params: int,
    result: SplitResult,
    history: dict,
    val_metrics: dict,
    test_metrics: dict,
) -> plt.Figure:
    arch = cfg["model"]["architecture"]

    fig = plt.figure(figsize=A4_P, facecolor="#1a1a2e")

    def _t(x, y, s, **kw):
        fig.text(x, y, s, transform=fig.transFigure, ha="center", **kw)

    _t(0.5, 0.92, "Experiment 1", fontsize=30, color="white", fontweight="bold")
    _t(0.5, 0.87, "INR Baseline — Dense Training · Uniform Validation",
       fontsize=13, color="#a0c4ff")
    _t(0.5, 0.84, "SIREN backbone · data loss only · no physics",
       fontsize=10, color="#8899bb")

    line_kw = dict(transform=fig.transFigure, color="#334466", lw=0.8)
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.82, 0.82], **line_kw))

    # Two-column info section
    ax = fig.add_axes([0.08, 0.32, 0.84, 0.48])
    ax.axis("off")

    left_lines = [
        ("Model", f"SIREN  {arch['hidden_dim']} × {arch['n_layers']} layers"),
        ("ω₀",   f"{arch['omega_0']}"),
        ("Parameters", f"{n_params:,}"),
        ("",     ""),
        ("Split mode",   "uniform"),
        ("Train",  f"{result.info['actual_train_fraction']:.1%}  "
                   f"({result.info['n_train_profiles']:,} profiles)"),
        ("Validation", f"{result.info['actual_val_fraction']:.1%}  "
                       f"({result.info['n_val_profiles']:,} profiles)"),
        ("Test",   f"{result.info['actual_test_fraction']:.1%}  "
                   f"({result.info['n_test_profiles']:,} profiles)"),
        ("Seed",  str(args.seed)),
    ]
    right_lines = [
        ("Learning rate",  f"{args.lr or cfg['training']['learning_rate']:.1e}"),
        ("Batch size",     f"{args.batch_size:,}"),
        ("Max epochs",     f"{args.epochs:,}"),
        ("Patience",       f"{args.patience}"),
        ("",     ""),
        ("Best epoch",     f"{history['best_epoch']}"),
        ("Best val loss",  f"{history['best_val_loss']:.6f}"),
        ("",     ""),
        ("Val  T RMSE",    f"{val_metrics['T_rmse']:.4f} °C"),
        ("Val  S RMSE",    f"{val_metrics['S_rmse']:.5f} PSU"),
        ("Test T RMSE",    f"{test_metrics['T_rmse']:.4f} °C"),
        ("Test S RMSE",    f"{test_metrics['S_rmse']:.5f} PSU"),
    ]

    y0, dy = 0.96, 0.085
    for i, (key, val) in enumerate(left_lines):
        y = y0 - i * dy
        ax.text(0.03, y, key,  fontsize=9.5, color="#aabbdd", va="top")
        ax.text(0.30, y, val,  fontsize=9.5, color="white",   va="top", fontweight="bold")

    for i, (key, val) in enumerate(right_lines):
        y = y0 - i * dy
        ax.text(0.55, y, key,  fontsize=9.5, color="#aabbdd", va="top")
        ax.text(0.78, y, val,  fontsize=9.5, color="white",   va="top", fontweight="bold")

    fig.add_artist(plt.Line2D([0.08, 0.92], [0.30, 0.30], **line_kw))
    _t(0.5, 0.26, f"Config: {args.config.name}   ·   Zarr: {args.zarr_path.name}",
       fontsize=8, color="#667799")
    _t(0.5, 0.22, datetime.now().strftime("%Y-%m-%d %H:%M"),
       fontsize=8, color="#667799")

    return fig


def pdf_spatial_fields(
    ds: xr.Dataset, var_temp: str, var_sal: str,
    label_T: str, label_S: str,
) -> plt.Figure:
    """A4 landscape — mean surface T (left) · mean surface S (right)."""
    lons = ds.longitude.values
    lats = ds.latitude.values
    T_surf = ds[var_temp].isel(depth=0).mean(dim="time").values
    S_surf = ds[var_sal].isel(depth=0).mean(dim="time").values

    fig, axes = plt.subplots(1, 2, figsize=A4_L)
    fig.suptitle("Mean Annual Sea Surface Fields", fontsize=13, fontweight="bold", y=0.98)

    _draw_surface_field(axes[0], lons, lats, T_surf, label_T, CMAP_T, f"Surface {label_T}")
    _draw_surface_field(axes[1], lons, lats, S_surf, label_S, CMAP_S, f"Surface {label_S}")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def pdf_temporal_profiles(
    ds: xr.Dataset, result: SplitResult, depths_g: np.ndarray,
    var_temp: str, var_sal: str, label_T: str, label_S: str,
) -> plt.Figure:
    """A4 portrait — time series (top) + vertical profiles + depth counts (bottom)."""
    times = pd.to_datetime(ds.time.values)
    sst   = ds[var_temp].isel(depth=0).mean(dim=["latitude", "longitude"]).values
    sss   = ds[var_sal].isel(depth=0).mean(dim=["latitude", "longitude"]).values
    T_prof = ds[var_temp].mean(dim=["latitude", "longitude", "time"]).values
    S_prof = ds[var_sal].mean(dim=["latitude", "longitude", "time"]).values

    fig = plt.figure(figsize=A4_P)
    gs  = fig.add_gridspec(3, 3,
                            height_ratios=[1.4, 1.4, 4],
                            hspace=0.52, wspace=0.38,
                            left=0.10, right=0.96,
                            top=0.95,  bottom=0.05)

    ax_tsT = fig.add_subplot(gs[0, :])
    ax_tsS = fig.add_subplot(gs[1, :], sharex=ax_tsT)
    ax_prT = fig.add_subplot(gs[2, 0])
    ax_dep = fig.add_subplot(gs[2, 1], sharey=ax_prT)
    ax_prS = fig.add_subplot(gs[2, 2], sharey=ax_prT)

    _draw_timeseries(ax_tsT, ax_tsS, times, sst, sss, label_T, label_S)
    _draw_profile(ax_prT, T_prof, depths_g, label_T, COL_T)
    _draw_profile(ax_prS, S_prof, depths_g, label_S, COL_S)
    ax_prS.set_ylabel("")
    ax_prS.tick_params(labelleft=False)
    _draw_depth_counts(ax_dep, result.i_dep, depths_g)
    ax_dep.set_ylabel("")
    ax_dep.tick_params(labelleft=False)

    fig.suptitle("Temporal & Vertical Distribution", fontsize=12, fontweight="bold")
    return fig


def pdf_split_map(result: SplitResult, ds: xr.Dataset) -> plt.Figure:
    """A4 landscape — train / validation / test fraction maps (1×3)."""
    lons_g = ds.longitude.values
    lats_g = ds.latitude.values
    n_lon, n_lat = len(lons_g), len(lats_g)

    cell_id = (result.i_lat * n_lon + result.i_lon).astype(np.int64)
    tot = np.maximum(np.bincount(cell_id, minlength=n_lon * n_lat).reshape(n_lon, n_lat), 1)

    def _frac(mask):
        return np.bincount(cell_id[mask], minlength=n_lon * n_lat).reshape(n_lon, n_lat) / tot

    frac_tr = _frac(result.train_mask)
    frac_va = _frac(result.val_mask)
    frac_te = _frac(result.test_mask)

    fig, axes = plt.subplots(1, 3, figsize=A4_L)
    info = result.info
    fig.suptitle(
        f"Profile Split — uniform mode  ·  seed={info['seed']}  ·  "
        f"train {info['actual_train_fraction']:.0%}  "
        f"val {info['actual_val_fraction']:.0%}  "
        f"test {info['actual_test_fraction']:.0%}",
        fontsize=11, fontweight="bold", y=0.99,
    )

    _draw_split_frac(axes[0], lons_g, lats_g, frac_tr, "Train fraction",      "Blues")
    _draw_split_frac(axes[1], lons_g, lats_g, frac_va, "Validation fraction", "Oranges")
    _draw_split_frac(axes[2], lons_g, lats_g, frac_te, "Test fraction",       "Greens")

    for ax in axes[1:]:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def pdf_training(history: dict) -> plt.Figure:
    """A4 portrait — 4 stacked subplots sharing the epoch x-axis."""
    ep      = history["epoch"]
    best_ep = history["best_epoch"]

    fig, axes = plt.subplots(4, 1, figsize=A4_P, sharex=True)
    fig.suptitle("Training History", fontsize=13, fontweight="bold")

    _draw_loss_curve(axes[0], ep, history["train"],   history["val"],
                     "Total MSE (CT + SA)", best_ep)
    _draw_loss_curve(axes[1], ep, history["train_T"], history["val_T"],
                     "Temperature MSE", best_ep, COL_T, "#e8836e")
    _draw_loss_curve(axes[2], ep, history["train_S"], history["val_S"],
                     "Salinity MSE",    best_ep, COL_S, "#7ab8d4")
    _draw_lr_curve(axes[3], ep, history["lr"])

    axes[-1].set_xlabel("Epoch")

    # Minimal x-tick overlap between panels
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=0.6)
    return fig


def pdf_evaluation(
    split_name: str,
    metrics: dict,
    depths_g: np.ndarray,
    label_T: str,
    label_S: str,
) -> plt.Figure:
    """A4 portrait — 2×2: scatter plots (top row) + RMSE vs depth (bottom row)."""
    T_pred, S_pred = metrics["T_pred"], metrics["S_pred"]
    T_true, S_true = metrics["T_true"], metrics["S_true"]
    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=A4_P)
    fig.suptitle(
        f"{split_name} — {metrics['n_obs']:,} observations",
        fontsize=13, fontweight="bold",
    )
    gs = fig.add_gridspec(2, 2,
                           height_ratios=[3, 4],
                           hspace=0.42, wspace=0.32,
                           left=0.10, right=0.96,
                           top=0.94,  bottom=0.06)

    ax_sT = fig.add_subplot(gs[0, 0])
    ax_sS = fig.add_subplot(gs[0, 1])
    ax_dT = fig.add_subplot(gs[1, 0])
    ax_dS = fig.add_subplot(gs[1, 1], sharey=ax_dT)

    _draw_scatter(ax_sT, T_pred, T_true, label_T, "°C",
                  metrics["T_rmse"], metrics["T_mae"], COL_T, rng)
    _draw_scatter(ax_sS, S_pred, S_true, label_S, "PSU",
                  metrics["S_rmse"], metrics["S_mae"], COL_S, rng)

    T_rmse_d = np.array(metrics["T_rmse_depth"])
    S_rmse_d = np.array(metrics["S_rmse_depth"])
    _draw_rmse_depth(ax_dT, T_rmse_d, depths_g, label_T, "°C",  COL_T)
    _draw_rmse_depth(ax_dS, S_rmse_d, depths_g, label_S, "PSU", COL_S)
    ax_dS.set_ylabel("")
    ax_dS.tick_params(labelleft=False)

    return fig


def pdf_summary(
    val_m: dict,
    test_m: dict,
    label_T: str,
    label_S: str,
    history: dict,
    n_params: int,
) -> plt.Figure:
    """A4 portrait — metrics table + side-by-side RMSE-vs-depth for val and test."""
    fig = plt.figure(figsize=A4_P)
    fig.suptitle("Summary — Final Metrics", fontsize=13, fontweight="bold")
    gs = fig.add_gridspec(3, 2,
                           height_ratios=[1.8, 4, 4],
                           hspace=0.55, wspace=0.38,
                           left=0.10, right=0.96,
                           top=0.94,  bottom=0.05)

    # ── Metrics table ─────────────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[0, :])
    ax_tbl.axis("off")
    rows = [
        ["Validation",
         f"{val_m['T_rmse']:.4f}", f"{val_m['T_mae']:.4f}",
         f"{val_m['S_rmse']:.5f}", f"{val_m['S_mae']:.5f}"],
        ["Test",
         f"{test_m['T_rmse']:.4f}", f"{test_m['T_mae']:.4f}",
         f"{test_m['S_rmse']:.5f}", f"{test_m['S_mae']:.5f}"],
    ]
    cols = ["Set", f"{label_T}\nRMSE", f"{label_T}\nMAE",
            f"{label_S}\nRMSE", f"{label_S}\nMAE"]
    tbl = ax_tbl.table(cellText=rows, colLabels=cols,
                       cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.4, 2.2)
    for (r, _), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 1:
            cell.set_facecolor("#eaf1fb")

    # ── RMSE-vs-depth: val and test side by side ──────────────────────────────
    ax_vT = fig.add_subplot(gs[1, 0])
    ax_tT = fig.add_subplot(gs[1, 1])
    # Salinity
    ax_vS = fig.add_subplot(gs[2, 0])
    ax_tS = fig.add_subplot(gs[2, 1])

    _draw_rmse_depth(ax_vT, np.array(val_m["T_rmse_depth"]),
                     np.arange(len(val_m["T_rmse_depth"])), label_T, "°C",  COL_T)
    _draw_rmse_depth(ax_tT, np.array(test_m["T_rmse_depth"]),
                     np.arange(len(test_m["T_rmse_depth"])), label_T, "°C",  COL_T)
    _draw_rmse_depth(ax_vS, np.array(val_m["S_rmse_depth"]),
                     np.arange(len(val_m["S_rmse_depth"])), label_S, "PSU", COL_S)
    _draw_rmse_depth(ax_tS, np.array(test_m["S_rmse_depth"]),
                     np.arange(len(test_m["S_rmse_depth"])), label_S, "PSU", COL_S)

    for ax in (ax_vT, ax_tT, ax_vS, ax_tS):
        ax.set_ylabel("Depth level index")

    ax_vT.set_title(f"Validation — {label_T} RMSE", fontsize=9)
    ax_tT.set_title(f"Test — {label_T} RMSE",       fontsize=9)
    ax_vS.set_title(f"Validation — {label_S} RMSE", fontsize=9)
    ax_tS.set_title(f"Test — {label_S} RMSE",       fontsize=9)

    fig.text(0.5, 0.01,
             f"Parameters: {n_params:,}   Best epoch: {history['best_epoch']}   "
             f"Best val loss: {history['best_val_loss']:.6f}",
             ha="center", fontsize=8, color="#555")

    return fig


# ── Individual PNG savers ─────────────────────────────────────────────────────

def save_individual_pngs(
    ds: xr.Dataset,
    result: SplitResult,
    depths_g: np.ndarray,
    var_temp: str,
    var_sal: str,
    label_T: str,
    label_S: str,
    history: dict,
    val_metrics: dict,
    test_metrics: dict,
    plots_dir: Path,
) -> list[Path]:
    saved: list[Path] = []
    lons_g = ds.longitude.values
    lats_g = ds.latitude.values
    times  = pd.to_datetime(ds.time.values)
    rng    = np.random.default_rng(0)

    # Surface fields
    for var, label, cmap, name in [
        (var_temp, label_T, CMAP_T, "dist_surface_T"),
        (var_sal,  label_S, CMAP_S, "dist_surface_S"),
    ]:
        data = ds[var].isel(depth=0).mean(dim="time").values
        fig, ax = plt.subplots(figsize=(7, 5))
        _draw_surface_field(ax, lons_g, lats_g, data, label, cmap, f"Mean surface {label}")
        fig.tight_layout()
        saved.append(_save_fig(fig, plots_dir, name))

    # Time series
    sst = ds[var_temp].isel(depth=0).mean(dim=["latitude", "longitude"]).values
    sss = ds[var_sal].isel(depth=0).mean(dim=["latitude", "longitude"]).values
    fig, (ax_T, ax_S) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    _draw_timeseries(ax_T, ax_S, times, sst, sss, label_T, label_S)
    fig.tight_layout()
    saved.append(_save_fig(fig, plots_dir, "dist_timeseries"))

    # Vertical profiles
    T_prof = ds[var_temp].mean(dim=["latitude", "longitude", "time"]).values
    S_prof = ds[var_sal].mean(dim=["latitude", "longitude", "time"]).values
    fig, (ax_T, ax_S) = plt.subplots(1, 2, figsize=(8, 7))
    _draw_profile(ax_T, T_prof, depths_g, label_T, COL_T)
    _draw_profile(ax_S, S_prof, depths_g, label_S, COL_S)
    fig.tight_layout()
    saved.append(_save_fig(fig, plots_dir, "dist_vertical_profiles"))

    # Depth counts
    fig, ax = plt.subplots(figsize=(5, 8))
    _draw_depth_counts(ax, result.i_dep, depths_g)
    fig.tight_layout()
    saved.append(_save_fig(fig, plots_dir, "dist_depth_counts"))

    # Training curves — individual files for each loss
    ep      = history["epoch"]
    best_ep = history["best_epoch"]
    for key_tr, key_va, title, col_tr, col_va, name in [
        ("train",   "val",   "Total MSE",       COL_TR, COL_VA, "train_loss_total"),
        ("train_T", "val_T", "Temperature MSE", COL_T,  "#e8836e", "train_loss_T"),
        ("train_S", "val_S", "Salinity MSE",    COL_S,  "#7ab8d4", "train_loss_S"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4))
        _draw_loss_curve(ax, ep, history[key_tr], history[key_va],
                         title, best_ep, col_tr, col_va)
        ax.set_xlabel("Epoch")
        fig.tight_layout()
        saved.append(_save_fig(fig, plots_dir, name))

    fig, ax = plt.subplots(figsize=(9, 3))
    _draw_lr_curve(ax, ep, history["lr"])
    ax.set_xlabel("Epoch")
    fig.tight_layout()
    saved.append(_save_fig(fig, plots_dir, "train_lr"))

    # Evaluation scatter + RMSE-vs-depth — individual files
    for split_name, m in [("val", val_metrics), ("test", test_metrics)]:
        tag = split_name
        T_pred, S_pred = m["T_pred"], m["S_pred"]
        T_true, S_true = m["T_true"], m["S_true"]
        T_rmse_d = np.array(m["T_rmse_depth"])
        S_rmse_d = np.array(m["S_rmse_depth"])

        for pred, true, label, unit, rmse, mae, col, name in [
            (T_pred, T_true, label_T, "°C",  m["T_rmse"], m["T_mae"], COL_T, f"{tag}_scatter_T"),
            (S_pred, S_true, label_S, "PSU", m["S_rmse"], m["S_mae"], COL_S, f"{tag}_scatter_S"),
        ]:
            fig, ax = plt.subplots(figsize=(6, 6))
            _draw_scatter(ax, pred, true, label, unit, rmse, mae, col, rng)
            ax.set_title(f"{split_name.capitalize()} — {label}\n"
                         f"RMSE = {rmse:.4f} {unit}   MAE = {mae:.4f} {unit}")
            fig.tight_layout()
            saved.append(_save_fig(fig, plots_dir, name))

        for rmse_d, label, unit, col, name in [
            (T_rmse_d, label_T, "°C",  COL_T, f"{tag}_rmse_depth_T"),
            (S_rmse_d, label_S, "PSU", COL_S, f"{tag}_rmse_depth_S"),
        ]:
            fig, ax = plt.subplots(figsize=(5, 8))
            _draw_rmse_depth(ax, rmse_d, depths_g, label, unit, col)
            ax.set_title(f"{split_name.capitalize()} — {label} RMSE vs depth")
            fig.tight_layout()
            saved.append(_save_fig(fig, plots_dir, name))

    return saved


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    model: INR,
    train_coords: torch.Tensor,
    train_targets: torch.Tensor,
    val_coords: torch.Tensor,
    val_targets: torch.Tensor,
    args: argparse.Namespace,
    cfg: dict,
    ckpt_dir: Path,
    device: torch.device,
) -> tuple[dict, dict, int]:
    lr      = args.lr or cfg["training"]["learning_rate"]
    n_train = train_coords.shape[0]
    n_val   = val_coords.shape[0]

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=lr * 1e-2,
    )

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
    infer_batch  = args.infer_batch

    pbar = trange(1, args.epochs + 1, desc="Training", unit="ep", dynamic_ncols=True)

    for epoch in pbar:
        model.train()
        ep_tot = ep_T = ep_S = 0.0
        perm = torch.randperm(n_train, device=device)

        for start in range(0, n_train, args.batch_size):
            xb = train_coords[perm[start : start + args.batch_size]]
            yb = train_targets[perm[start : start + args.batch_size]]
            loss, per = model.loss(xb, yb)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            n = xb.shape[0]
            ep_tot += loss.detach().item() * n
            ep_T   += per["CT"].detach().item() * n
            ep_S   += per["SA"].detach().item() * n

        ep_tot /= n_train
        ep_T   /= n_train
        ep_S   /= n_train
        scheduler.step()

        model.eval()
        v_tot = v_T = v_S = 0.0
        with torch.inference_mode():
            for start in range(0, n_val, infer_batch):
                xb = val_coords[start : start + infer_batch]
                yb = val_targets[start : start + infer_batch]
                _, per = model.loss(xb, yb)
                n = xb.shape[0]
                v_tot += (per["CT"].item() + per["SA"].item()) * n
                v_T   += per["CT"].item() * n
                v_S   += per["SA"].item() * n
        v_tot /= n_val
        v_T   /= n_val
        v_S   /= n_val

        history["epoch"].append(epoch)
        history["lr"].append(scheduler.get_last_lr()[0])
        history["train"].append(ep_tot); history["train_T"].append(ep_T); history["train_S"].append(ep_S)
        history["val"].append(v_tot);   history["val_T"].append(v_T);   history["val_S"].append(v_S)

        if v_tot < best_val - args.min_delta:
            best_val   = v_tot
            best_epoch = epoch
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= args.patience:
            stopped = epoch
            pbar.write(f"[epoch {epoch}] Early stop — best val={best_val:.6f} @ epoch {best_epoch}")
            break

        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_loss": v_tot, "best_val": best_val},
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


# ── Inference + metrics ───────────────────────────────────────────────────────

@torch.inference_mode()
def infer_split(
    model: INR,
    coords_np: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    infer_batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    coords = torch.from_numpy(coords_np[mask]).to(device)
    n  = coords.shape[0]
    T_ = np.empty(n, dtype=np.float32)
    S_ = np.empty(n, dtype=np.float32)
    model.eval()
    for start in range(0, n, infer_batch):
        end = min(start + infer_batch, n)
        out = model(coords[start:end])
        T_[start:end] = out["CT"].cpu().numpy()
        S_[start:end] = out["SA"].cpu().numpy()
    return T_, S_


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

    n_dep  = len(depths_g)
    T_rd   = np.full(n_dep, np.nan)
    S_rd   = np.full(n_dep, np.nan)
    T_md   = np.full(n_dep, np.nan)
    S_md   = np.full(n_dep, np.nan)

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


# ── Save results.json ─────────────────────────────────────────────────────────

def save_results(
    args: argparse.Namespace,
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
        "experiment":  "experiment1",
        "run_time":    datetime.now().isoformat(),
        "config_file": str(args.config),
        "zarr_path":   str(args.zarr_path),
        "seed":        args.seed,
        "split": {
            "mode": "uniform",
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


# ── Build PDF ─────────────────────────────────────────────────────────────────

def build_pdf(pages: list[plt.Figure], output_dir: Path) -> Path:
    """Write all A4 page figures to a single PDF and close each figure."""
    pdf_path = output_dir / "report.pdf"
    with PdfPages(pdf_path) as pdf:
        for fig in pages:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        pdf.infodict()["Title"]   = "Experiment 1 — INR Baseline"
        pdf.infodict()["Subject"] = "Ocean INR reconstruction — GLORYS12V1 patch 34S 69E"
    return pdf_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    cfg    = yaml.safe_load(open(args.config))
    device = get_device(args.device)
    bounds = cfg["normalisation"]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = args.output_dir
    plots_dir  = output_dir / "plots"
    ckpt_dir   = output_dir / "checkpoints"
    for d in (output_dir, plots_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("─" * 60)
    print("Experiment 1 — INR  |  uniform split")
    print("─" * 60)
    print(f"  Device  : {device}")
    print(f"  Config  : {args.config.name}")
    print(f"  Zarr    : {args.zarr_path}")
    print(f"  Output  : {output_dir}")
    print()

    # ── Load dataset ──────────────────────────────────────────────────────────
    t0 = time.time()
    ds = load_dataset(args.zarr_path, args.zarr_group)
    depths_g = ds.depth.values
    print(f"Dataset loaded ({time.time()-t0:.1f}s): {dict(ds.sizes)}")

    # ── Split profiles ────────────────────────────────────────────────────────
    print("\nBuilding profile split …")
    splitter = GlorysProfileSplitter(ds, variables=(args.var_temp, args.var_sal))
    result   = splitter.split(
        mode="uniform",
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    splitter.print_summary(result)

    # ── Normalised arrays ─────────────────────────────────────────────────────
    print("\nBuilding coordinate and target arrays …", flush=True)
    t0 = time.time()
    coords_np, targets_np, targets_phys = build_arrays(
        result, ds, args.var_temp, args.var_sal, bounds
    )
    print(f"  Done ({time.time()-t0:.1f}s)  coords={coords_np.shape}  targets={targets_np.shape}")

    def _to_device(mask):
        return (torch.from_numpy(coords_np[mask]).to(device),
                torch.from_numpy(targets_np[mask]).to(device))

    train_coords, train_targets = _to_device(result.train_mask)
    val_coords,   val_targets   = _to_device(result.val_mask)

    # ── Build model ───────────────────────────────────────────────────────────
    model    = INR.from_config(cfg).to(device)
    n_params = model.n_parameters()
    print(f"\nModel: {model}")
    print(f"Parameters: {n_params:,}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nStarting training …")
    t0 = time.time()
    history, best_state, best_epoch = train(
        model, train_coords, train_targets,
        val_coords, val_targets,
        args, cfg, ckpt_dir, device,
    )
    train_elapsed = time.time() - t0
    print(f"\nTraining done in {train_elapsed/60:.1f} min")
    print(f"  Best epoch : {best_epoch}")
    print(f"  Best val   : {history['best_val_loss']:.6f}")

    torch.save(
        {"model_state": best_state, "epoch": best_epoch,
         "val_loss": history["best_val_loss"], "cfg": cfg,
         "split_info": result.info},
        output_dir / "best_model.pt",
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    model.load_state_dict(best_state)

    print("\nEvaluating on validation set …", flush=True)
    T_vp, S_vp = infer_split(model, coords_np, result.val_mask, device, args.infer_batch)
    val_metrics = compute_metrics(T_vp, S_vp,
                                  targets_phys[result.val_mask],
                                  result.i_dep[result.val_mask], depths_g, bounds)
    print(f"  T RMSE: {val_metrics['T_rmse']:.4f}°C   S RMSE: {val_metrics['S_rmse']:.5f} PSU")

    print("Evaluating on test set …", flush=True)
    T_tp, S_tp = infer_split(model, coords_np, result.test_mask, device, args.infer_batch)
    test_metrics = compute_metrics(T_tp, S_tp,
                                   targets_phys[result.test_mask],
                                   result.i_dep[result.test_mask], depths_g, bounds)
    print(f"  T RMSE: {test_metrics['T_rmse']:.4f}°C   S RMSE: {test_metrics['S_rmse']:.5f} PSU")

    # ── Save results.json ─────────────────────────────────────────────────────
    results_path = save_results(args, cfg, result.info, history,
                                val_metrics, test_metrics, output_dir)
    print(f"\nResults saved → {results_path}")

    # ── Individual PNGs ───────────────────────────────────────────────────────
    print("Saving individual plot files …", flush=True)
    save_individual_pngs(
        ds, result, depths_g, args.var_temp, args.var_sal,
        args.label_temp, args.label_sal,
        history, val_metrics, test_metrics, plots_dir,
    )

    # ── Build PDF (A4 composite pages) ────────────────────────────────────────
    print("Building PDF report …", flush=True)
    pages = [
        pdf_cover(args, cfg, n_params, result, history, val_metrics, test_metrics),
        pdf_spatial_fields(ds, args.var_temp, args.var_sal,
                           args.label_temp, args.label_sal),
        pdf_temporal_profiles(ds, result, depths_g,
                              args.var_temp, args.var_sal,
                              args.label_temp, args.label_sal),
        pdf_split_map(result, ds),
        pdf_training(history),
        pdf_evaluation("Validation", val_metrics, depths_g,
                       args.label_temp, args.label_sal),
        pdf_evaluation("Test", test_metrics, depths_g,
                       args.label_temp, args.label_sal),
        pdf_summary(val_metrics, test_metrics,
                    args.label_temp, args.label_sal, history, n_params),
    ]
    pdf_path = build_pdf(pages, output_dir)
    print(f"PDF report   → {pdf_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print("─" * 60)
    print("Experiment 1 complete")
    print("─" * 60)
    print(f"  Stopped epoch   : {history['stopped_epoch']}")
    print(f"  Best val loss   : {history['best_val_loss']:.6f}  (epoch {best_epoch})")
    print(f"  Val  T RMSE     : {val_metrics['T_rmse']:.4f} °C")
    print(f"  Val  S RMSE     : {val_metrics['S_rmse']:.5f} PSU")
    print(f"  Test T RMSE     : {test_metrics['T_rmse']:.4f} °C")
    print(f"  Test S RMSE     : {test_metrics['S_rmse']:.5f} PSU")
    print(f"  Parameters      : {n_params:,}")
    print(f"  Training time   : {train_elapsed/60:.1f} min")
    print(f"  Output dir      : {output_dir}")
    print("─" * 60)


if __name__ == "__main__":
    main()
