"""PDF report and PNG plotting for INR experiments.

All matplotlib/PDF logic lives here so runner.py stays focused on
data loading, training, and evaluation.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from inrpinn.data.splitter import SplitResult

# ── A4 dimensions & colour palette ───────────────────────────────────────────

A4_P = (8.27, 11.69)   # portrait  (inches)
A4_L = (11.69, 8.27)   # landscape (inches)

CMAP_T = "RdYlBu_r"
CMAP_S = "viridis"
COL_T  = "#c0392b"
COL_S  = "#2980b9"
COL_TR = "#2c7bb6"
COL_VA = "#d7191c"

plt.rcParams.update({
    "font.size":       9,
    "axes.titlesize": 10,
    "axes.labelsize":  9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid":      True,
    "grid.alpha":     0.35,
})


# ── Low-level draw helpers ────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, plots_dir: Path, name: str) -> Path:
    p = plots_dir / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


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
    n   = min(50_000, len(pred))
    idx = rng.choice(len(pred), n, replace=False)
    ax.scatter(true[idx], pred[idx], s=2, alpha=0.25, color=color,
               edgecolors="none", rasterized=True)
    lo = min(true.min(), pred.min())
    hi = max(true.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="1 : 1")
    ax.set_xlabel(f"GLORYS {label}")
    ax.set_ylabel(f"INR {label}")
    ax.set_title(f"{label}\nRMSE = {rmse:.4f} {unit}   MAE = {mae:.4f} {unit}")
    ax.legend(fontsize=7)


def _draw_rmse_depth(ax, rmse_d, depths, label, unit, color):
    valid = ~np.isnan(rmse_d)
    ax.plot(rmse_d[valid], depths[valid], "o-", color=color, ms=3, lw=1.4)
    ax.invert_yaxis()
    ax.set_xlabel(f"RMSE ({unit})")
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"{label} RMSE vs depth\nOverall = {np.nanmean(rmse_d):.4f} {unit}")


# ── PDF page builders ─────────────────────────────────────────────────────────

def pdf_cover(
    exp_name: str,
    val_mode_label: str,
    args,
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

    _t(0.5, 0.92, exp_name,      fontsize=30, color="white", fontweight="bold")
    _t(0.5, 0.87, f"INR Baseline — Dense Training · {val_mode_label}",
       fontsize=13, color="#a0c4ff")
    _t(0.5, 0.84, "SIREN backbone · data loss only · no physics",
       fontsize=10, color="#8899bb")

    line_kw = dict(transform=fig.transFigure, color="#334466", lw=0.8)
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.82, 0.82], **line_kw))

    ax = fig.add_axes([0.08, 0.32, 0.84, 0.48])
    ax.axis("off")

    left_lines = [
        ("Model",      f"SIREN  {arch['hidden_dim']} × {arch['n_layers']} layers"),
        ("ω₀",         f"{arch['omega_0']}"),
        ("Parameters", f"{n_params:,}"),
        ("", ""),
        ("Val mode",   result.info["mode"]),
        ("Train",      f"{result.info['actual_train_fraction']:.1%}  "
                       f"({result.info['n_train_profiles']:,} profiles)"),
        ("Validation", f"{result.info['actual_val_fraction']:.1%}  "
                       f"({result.info['n_val_profiles']:,} profiles)"),
        ("Test",       f"{result.info['actual_test_fraction']:.1%}  "
                       f"({result.info['n_test_profiles']:,} profiles)"),
        ("Seed",       str(args.seed)),
    ]
    right_lines = [
        ("Learning rate", f"{args.lr or cfg['training']['learning_rate']:.1e}"),
        ("Batch size",    f"{args.batch_size:,}"),
        ("Max epochs",    f"{args.epochs:,}"),
        ("Patience",      f"{args.patience}"),
        ("", ""),
        ("Best epoch",    f"{history['best_epoch']}"),
        ("Best val loss", f"{history['best_val_loss']:.6f}"),
        ("", ""),
        ("Val  T RMSE",   f"{val_metrics['T_rmse']:.4f} °C"),
        ("Val  S RMSE",   f"{val_metrics['S_rmse']:.5f} PSU"),
        ("Test T RMSE",   f"{test_metrics['T_rmse']:.4f} °C"),
        ("Test S RMSE",   f"{test_metrics['S_rmse']:.5f} PSU"),
    ]

    y0, dy = 0.96, 0.085
    for i, (key, val) in enumerate(left_lines):
        y = y0 - i * dy
        ax.text(0.03, y, key, fontsize=9.5, color="#aabbdd", va="top")
        ax.text(0.30, y, val, fontsize=9.5, color="white",   va="top", fontweight="bold")

    for i, (key, val) in enumerate(right_lines):
        y = y0 - i * dy
        ax.text(0.55, y, key, fontsize=9.5, color="#aabbdd", va="top")
        ax.text(0.78, y, val, fontsize=9.5, color="white",   va="top", fontweight="bold")

    fig.add_artist(plt.Line2D([0.08, 0.92], [0.30, 0.30], **line_kw))
    _t(0.5, 0.26, f"Config: {args.config.name}   ·   Zarr: {args.zarr_path.name}",
       fontsize=8, color="#667799")
    _t(0.5, 0.22, datetime.now().strftime("%Y-%m-%d %H:%M"),
       fontsize=8, color="#667799")

    return fig


def pdf_spatial_fields(
    ds, var_temp: str, var_sal: str,
    label_T: str, label_S: str,
) -> plt.Figure:
    lons    = ds.longitude.values
    lats    = ds.latitude.values
    T_surf  = ds[var_temp].isel(depth=0).mean(dim="time").values
    S_surf  = ds[var_sal].isel(depth=0).mean(dim="time").values

    fig, axes = plt.subplots(1, 2, figsize=A4_L)
    fig.suptitle("Mean Annual Sea Surface Fields", fontsize=13, fontweight="bold", y=0.98)

    _draw_surface_field(axes[0], lons, lats, T_surf, label_T, CMAP_T, f"Surface {label_T}")
    _draw_surface_field(axes[1], lons, lats, S_surf, label_S, CMAP_S, f"Surface {label_S}")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def pdf_temporal_profiles(
    ds, result: SplitResult, depths_g: np.ndarray,
    var_temp: str, var_sal: str, label_T: str, label_S: str,
) -> plt.Figure:
    times  = pd.to_datetime(ds.time.values)
    sst    = ds[var_temp].isel(depth=0).mean(dim=["latitude", "longitude"]).values
    sss    = ds[var_sal].isel(depth=0).mean(dim=["latitude", "longitude"]).values
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


def pdf_split_map(result: SplitResult, ds) -> plt.Figure:
    lons_g  = ds.longitude.values
    lats_g  = ds.latitude.values
    n_lon   = len(lons_g)
    n_lat   = len(lats_g)

    cell_id = (result.i_lat * n_lon + result.i_lon).astype(np.int64)
    tot     = np.maximum(
        np.bincount(cell_id, minlength=n_lon * n_lat).reshape(n_lon, n_lat), 1
    )

    def _frac(mask):
        return (
            np.bincount(cell_id[mask], minlength=n_lon * n_lat).reshape(n_lon, n_lat) / tot
        )

    info = result.info
    fig, axes = plt.subplots(1, 3, figsize=A4_L)
    fig.suptitle(
        f"Profile Split — {info['mode']} mode  ·  seed={info['seed']}  ·  "
        f"train {info['actual_train_fraction']:.0%}  "
        f"val {info['actual_val_fraction']:.0%}  "
        f"test {info['actual_test_fraction']:.0%}",
        fontsize=11, fontweight="bold", y=0.99,
    )

    _draw_split_frac(axes[0], lons_g, lats_g, _frac(result.train_mask), "Train",      "Blues")
    _draw_split_frac(axes[1], lons_g, lats_g, _frac(result.val_mask),   "Validation", "Oranges")
    _draw_split_frac(axes[2], lons_g, lats_g, _frac(result.test_mask),  "Test",       "Greens")

    for ax in axes[1:]:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def pdf_training(history: dict) -> plt.Figure:
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
    T_pred, S_pred = metrics["T_pred"], metrics["S_pred"]
    T_true, S_true = metrics["T_true"], metrics["S_true"]
    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=A4_P)
    fig.suptitle(f"{split_name} — {metrics['n_obs']:,} observations",
                 fontsize=13, fontweight="bold")
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
    _draw_rmse_depth(ax_dT, np.array(metrics["T_rmse_depth"]), depths_g, label_T, "°C",  COL_T)
    _draw_rmse_depth(ax_dS, np.array(metrics["S_rmse_depth"]), depths_g, label_S, "PSU", COL_S)
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
    fig = plt.figure(figsize=A4_P)
    fig.suptitle("Summary — Final Metrics", fontsize=13, fontweight="bold")
    gs = fig.add_gridspec(3, 2,
                           height_ratios=[1.8, 4, 4],
                           hspace=0.55, wspace=0.38,
                           left=0.10, right=0.96,
                           top=0.94,  bottom=0.05)

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

    ax_vT = fig.add_subplot(gs[1, 0])
    ax_tT = fig.add_subplot(gs[1, 1])
    ax_vS = fig.add_subplot(gs[2, 0])
    ax_tS = fig.add_subplot(gs[2, 1])

    idx_T = np.arange(len(val_m["T_rmse_depth"]))
    idx_S = np.arange(len(val_m["S_rmse_depth"]))
    _draw_rmse_depth(ax_vT, np.array(val_m["T_rmse_depth"]),  idx_T, label_T, "°C",  COL_T)
    _draw_rmse_depth(ax_tT, np.array(test_m["T_rmse_depth"]), idx_T, label_T, "°C",  COL_T)
    _draw_rmse_depth(ax_vS, np.array(val_m["S_rmse_depth"]),  idx_S, label_S, "PSU", COL_S)
    _draw_rmse_depth(ax_tS, np.array(test_m["S_rmse_depth"]), idx_S, label_S, "PSU", COL_S)

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
    ds,
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

    for var, label, cmap, name in [
        (var_temp, label_T, CMAP_T, "dist_surface_T"),
        (var_sal,  label_S, CMAP_S, "dist_surface_S"),
    ]:
        data = ds[var].isel(depth=0).mean(dim="time").values
        fig, ax = plt.subplots(figsize=(7, 5))
        _draw_surface_field(ax, lons_g, lats_g, data, label, cmap, f"Mean surface {label}")
        fig.tight_layout()
        saved.append(_save_fig(fig, plots_dir, name))

    sst = ds[var_temp].isel(depth=0).mean(dim=["latitude", "longitude"]).values
    sss = ds[var_sal].isel(depth=0).mean(dim=["latitude", "longitude"]).values
    fig, (ax_T, ax_S) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    _draw_timeseries(ax_T, ax_S, times, sst, sss, label_T, label_S)
    fig.tight_layout()
    saved.append(_save_fig(fig, plots_dir, "dist_timeseries"))

    T_prof = ds[var_temp].mean(dim=["latitude", "longitude", "time"]).values
    S_prof = ds[var_sal].mean(dim=["latitude", "longitude", "time"]).values
    fig, (ax_T, ax_S) = plt.subplots(1, 2, figsize=(8, 7))
    _draw_profile(ax_T, T_prof, depths_g, label_T, COL_T)
    _draw_profile(ax_S, S_prof, depths_g, label_S, COL_S)
    fig.tight_layout()
    saved.append(_save_fig(fig, plots_dir, "dist_vertical_profiles"))

    fig, ax = plt.subplots(figsize=(5, 8))
    _draw_depth_counts(ax, result.i_dep, depths_g)
    fig.tight_layout()
    saved.append(_save_fig(fig, plots_dir, "dist_depth_counts"))

    ep      = history["epoch"]
    best_ep = history["best_epoch"]
    for key_tr, key_va, title, col_tr, col_va, name in [
        ("train",   "val",   "Total MSE",       COL_TR,    COL_VA,    "train_loss_total"),
        ("train_T", "val_T", "Temperature MSE", COL_T,     "#e8836e", "train_loss_T"),
        ("train_S", "val_S", "Salinity MSE",    COL_S,     "#7ab8d4", "train_loss_S"),
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

    for split_name, m in [("val", val_metrics), ("test", test_metrics)]:
        tag      = split_name
        T_pred   = m["T_pred"];  S_pred   = m["S_pred"]
        T_true   = m["T_true"];  S_true   = m["S_true"]
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


# ── PDF assembly ──────────────────────────────────────────────────────────────

def build_pdf(pages: list[plt.Figure], output_dir: Path, title: str) -> Path:
    pdf_path = output_dir / "report.pdf"
    with PdfPages(pdf_path) as pdf:
        for fig in pages:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        pdf.infodict()["Title"]   = title
        pdf.infodict()["Subject"] = "Ocean INR reconstruction — GLORYS12V1"
    return pdf_path
