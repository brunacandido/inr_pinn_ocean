#!/usr/bin/env python3
"""Evaluate how well each of the 11 physics equations holds in GLORYS12V1.

All residuals are computed directly from GLORYS fields — no neural network.
The output guides which equations are safe PINN loss terms and at what weight.

Usage (CLI)
-----------
    python -m inrpinn.physics.diagnostics --store data/glorys_agulhas.zarr
    python -m inrpinn.physics.diagnostics --config configs/data_config.yaml

Usage (import)
--------------
    from inrpinn.physics.diagnostics import EquationDiagnostics
    diag = EquationDiagnostics("data/glorys_agulhas.zarr")
    diag.run_all()
    diag.save_summary()
"""
from __future__ import annotations

import argparse
import pathlib
import warnings
from dataclasses import dataclass

import gsw
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.integrate as sci_int
import xarray as xr
import yaml
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ── Physical constants ───────────────────────────────────────────────────────

G = 9.81            # m s⁻²
RHO0 = 1025.0       # kg m⁻³  reference density
R_EARTH = 6_371_000.0  # m
OMEGA = 7.292e-5    # rad s⁻¹  Earth rotation rate
CP_SEA = 3992.0     # J kg⁻¹ K⁻¹  specific heat of seawater

# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class EqResult:
    eq: str
    verdict: str
    mean_residual: float
    std_residual: float
    rmse: float
    p95: float
    recommended_weight: float
    notes: str


# ── Dataset validation result ────────────────────────────────────────────────

@dataclass
class DatasetResult:
    verdict: str          # "VALID" | "VALID_WITH_WARNINGS" | "INVALID"
    issues: list          # blocking problems that make physics meaningless
    warnings: list        # non-blocking concerns
    n_time: int
    n_depth: int
    time_range: tuple     # (str, str)
    lon_range: tuple      # (float, float)
    lat_range: tuple      # (float, float)
    depth_range: tuple    # (float, float)
    variable_stats: dict  # var → {nan_frac, mean, std, p5, p95}
    notes: str


# ── Physical plausibility ranges ─────────────────────────────────────────────

_PHYS_RANGES: dict[str, tuple[float, float, str]] = {
    "thetao": (-2.5,  35.0, "°C"),
    "so":     (20.0,  42.0, "PSU"),
    "uo":     (-5.0,   5.0, "m/s"),
    "vo":     (-5.0,   5.0, "m/s"),
    "zos":    (-3.0,   3.0, "m"),
    "mlotst": ( 0.5, 2000., "m"),
}

_REQUIRED_VARS = ["thetao", "so", "uo", "vo", "zos", "mlotst"]


# ── Finite-difference helpers ────────────────────────────────────────────────

def _d_dx(da: xr.DataArray) -> xr.DataArray:
    """d/dx in [da_units / m]: central differences along longitude."""
    lat_rad = np.deg2rad(da.latitude)
    m_per_deg = R_EARTH * np.cos(lat_rad) * (np.pi / 180.0)
    return da.differentiate("longitude") / m_per_deg


def _d_dy(da: xr.DataArray) -> xr.DataArray:
    """d/dy in [da_units / m]: central differences along latitude."""
    m_per_deg = R_EARTH * np.pi / 180.0
    return da.differentiate("latitude") / m_per_deg


def _d_dz(da: xr.DataArray) -> xr.DataArray:
    """d/dz in [da_units / m]: depth is positive downward."""
    return da.differentiate("depth")


def _d_dt(da: xr.DataArray) -> xr.DataArray:
    """d/dt in [da_units / s]: handles datetime64 coordinates."""
    return da.differentiate("time", datetime_unit="s")


def _coriolis(lat_da: xr.DataArray) -> xr.DataArray:
    """f = 2Ω sin(lat).  Negative in Southern Hemisphere."""
    return xr.DataArray(
        2.0 * OMEGA * np.sin(np.deg2rad(lat_da.values)),
        coords=lat_da.coords,
        dims=lat_da.dims,
    )


# ── Statistics helper ────────────────────────────────────────────────────────

def _nanmean_t(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """np.nanmean with the 'Mean of empty slice' RuntimeWarning suppressed.

    Land/bathymetry points are all-NaN for every time step, so nanmean over
    the time axis legitimately produces NaN there — the warning is expected.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        return np.nanmean(arr, axis=axis)


def _residual_stats(arr: np.ndarray, threshold: float | None = None) -> dict:
    flat = arr.ravel()
    flat = flat[np.isfinite(flat)]
    mean = float(np.mean(flat))
    std  = float(np.std(flat))
    rmse = float(np.sqrt(np.mean(flat**2)))
    p95  = float(np.percentile(np.abs(flat), 95))
    thr  = threshold if threshold is not None else std
    frac = float(np.mean(np.abs(flat) <= thr)) if len(flat) > 0 else np.nan
    return dict(mean=mean, std=std, rmse=rmse, p95=p95, fraction=frac)


# ── Map / figure helpers ─────────────────────────────────────────────────────

def _map_axes(n_cols: int = 2, n_rows: int = 1, figsize=None):
    if figsize is None:
        figsize = (7 * n_cols, 4.5 * n_rows)
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize,
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )
    ax_flat = np.array(axes).ravel() if n_rows * n_cols > 1 else [axes]
    for ax in ax_flat:
        ax.add_feature(cfeature.LAND, facecolor="#d0d0d0", zorder=3)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=4)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, linestyle="--",
                          color="gray", alpha=0.6)
        gl.top_labels = gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(range(15, 46, 10))
        gl.ylocator = mticker.FixedLocator(range(-45, -19, 5))
    return fig, np.array(axes), proj


def _savefig(fig: plt.Figure, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved → {path.name}")


# ── Verdict logic ────────────────────────────────────────────────────────────

def _verdict(rmse: float, thresholds: tuple[float, float, float]) -> str:
    """Map RMSE to a verdict string.

    thresholds: (use_max, caution_max, mask_max)
      rmse ≤ use_max      → 'USE'
      rmse ≤ mask_max     → 'USE_WITH_MASK'
      rmse ≤ caution_max  → 'USE_WITH_CAUTION'
      else                → 'DO_NOT_USE'
    """
    use_t, mask_t, caution_t = thresholds
    if rmse <= use_t:
        return "USE"
    if rmse <= mask_t:
        return "USE_WITH_MASK"
    if rmse <= caution_t:
        return "USE_WITH_CAUTION"
    return "DO_NOT_USE"


# ═══════════════════════════════════════════════════════════════════════════
# Main class
# ═══════════════════════════════════════════════════════════════════════════

class EquationDiagnostics:
    """Evaluate all 11 physics equations against GLORYS12V1 reanalysis.

    Parameters
    ----------
    store_path:
        Path to the Zarr store (data/glorys_agulhas.zarr).
        Loads from the ``raw`` group (thetao, so, uo, vo, zos, mlotst).
    out_dir:
        Directory for saved figures and CSV summary.
    kappa_T, kappa_S:
        Vertical diffusivity constants (m²/s) for EQ-9 and EQ-10.
    n_time:
        Number of time steps to use. None → all.  Reducing speeds up the run.
    """

    def __init__(
        self,
        store_path: str | pathlib.Path,
        out_dir: str | pathlib.Path = "results/diagnostics",
        kappa_T: float = 1e-5,
        kappa_S: float = 1e-5,
        n_time: int | None = None,
    ) -> None:
        self.store = pathlib.Path(store_path)
        self.out   = pathlib.Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.kappa_T = kappa_T
        self.kappa_S = kappa_S
        self.n_time  = n_time

        self._ds: xr.Dataset | None = None
        self._derived: dict = {}
        self._results: list[EqResult] = []

    # ── Data loading ─────────────────────────────────────────────────────────

    def _load(self) -> xr.Dataset:
        if self._ds is not None:
            return self._ds
        print("Loading GLORYS raw data …")
        ds = xr.open_zarr(str(self.store), group="raw", consolidated=False)
        if self.n_time is not None:
            ds = ds.isel(time=slice(None, self.n_time))
        # Rechunk so each spatial axis is one chunk.  The zarr chunk size (40)
        # does not evenly divide longitude (361), leaving a 1-element last chunk
        # that breaks xr.DataArray.differentiate (needs chunk > edge_order+1).
        ds = ds.chunk({"latitude": -1, "longitude": -1})
        self._ds = ds
        return ds

    def _derived_fields(self) -> dict:
        """Compute and cache SA, CT, rho, pressure, f, and diagnosed w."""
        if self._derived:
            return self._derived

        ds = self._load()
        print("Computing derived fields (SA, CT, ρ, p, f, w) …")

        # ── Pressure on the depth–lat grid (dbar; 1 dbar ≈ 1 m) ─────────────
        # gsw.p_from_z takes z positive upward, so z = -depth
        depth_v = ds.depth.values          # (nz,)
        lat_v   = ds.latitude.values       # (ny,)
        # broadcast to (nz, ny)
        p_2d = gsw.p_from_z(
            -depth_v[:, None] * np.ones((1, len(lat_v))),
            lat_v[None, :] * np.ones((len(depth_v), 1)),
        )  # (nz, ny)
        p_da = xr.DataArray(
            p_2d, coords={"depth": ds.depth, "latitude": ds.latitude},
            dims=["depth", "latitude"],
        )

        # ── SA, CT, rho (computed lazily on first .compute()) ────────────────
        sp = ds["so"]
        pt = ds["thetao"]

        # broadcast lon, lat, p to match 4-D field
        lon4 = ds.longitude
        lat4 = ds.latitude
        p4   = p_da  # xarray broadcasts (depth, lat) → (time, depth, lat, lon)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            SA = xr.apply_ufunc(
                gsw.SA_from_SP,
                sp, p4, lon4, lat4,
                dask="parallelized", output_dtypes=[float],
            )
            CT = xr.apply_ufunc(
                gsw.CT_from_pt,
                SA, pt,
                dask="parallelized", output_dtypes=[float],
            )
            rho = xr.apply_ufunc(
                gsw.rho,
                SA, CT, p4,
                dask="parallelized", output_dtypes=[float],
            )

        SA.name  = "SA"
        CT.name  = "CT"
        rho.name = "rho"

        # ── Coriolis ─────────────────────────────────────────────────────────
        f_da = xr.DataArray(
            2.0 * OMEGA * np.sin(np.deg2rad(lat_v)),
            coords={"latitude": ds.latitude}, dims=["latitude"],
        )

        # ── Diagnosed w from continuity ∂w/∂z = -(∂u/∂x + ∂v/∂y) ──────────
        # w(z) = -∫₀ᶻ (∂u/∂x + ∂v/∂y) dz,  w(surface) = 0
        div_h = _d_dx(ds["uo"]) + _d_dy(ds["vo"])  # (time, depth, lat, lon)

        # cumulative trapz over depth axis (axis=1 in 4-D array)
        dv = div_h.compute().values
        depth_v_full = ds.depth.values
        w_vals = -sci_int.cumulative_trapezoid(
            dv, x=depth_v_full, axis=1, initial=0.0
        )  # (time, ndepth, nlat, nlon)
        w_da = xr.DataArray(
            w_vals,
            coords=ds["uo"].coords, dims=ds["uo"].dims,
        )
        w_da.name = "wo"

        self._derived = dict(
            SA=SA, CT=CT, rho=rho, p=p_da, f=f_da, w=w_da, div_h=div_h,
        )
        return self._derived

    # ═══════════════════════════════════════════════════════════════════════
    # Dataset validation
    # ═══════════════════════════════════════════════════════════════════════

    def diagnose_dataset(self) -> DatasetResult:
        """Check that the Zarr store is readable, complete, and physically sane.

        Runs before any physics equation.  Returns a DatasetResult with verdict:
          VALID               — all checks pass
          VALID_WITH_WARNINGS — minor issues; physics diagnostics can still run
          INVALID             — blocking problem; physics diagnostics should be skipped
        """
        print("\n[DATASET]  Validating GLORYS store …")
        issues: list[str]    = []
        diag_warns: list[str] = []

        # ── 1. Open store ────────────────────────────────────────────────────
        try:
            ds = xr.open_zarr(str(self.store), group="raw", consolidated=False)
            if self.n_time is not None:
                ds = ds.isel(time=slice(None, self.n_time))
        except Exception as exc:
            return DatasetResult(
                verdict="INVALID",
                issues=[f"Cannot open Zarr store: {exc}"],
                warnings=[],
                n_time=0, n_depth=0,
                time_range=("N/A", "N/A"),
                lon_range=(np.nan, np.nan),
                lat_range=(np.nan, np.nan),
                depth_range=(np.nan, np.nan),
                variable_stats={},
                notes=f"Store unreadable — {exc}",
            )

        # ── 2. Required variables ────────────────────────────────────────────
        missing = [v for v in _REQUIRED_VARS if v not in ds.data_vars]
        if missing:
            issues.append(f"Missing required variables: {missing}")

        # ── 3. Dimension sizes ───────────────────────────────────────────────
        n_time  = ds.sizes.get("time",  0)
        n_depth = ds.sizes.get("depth", 0)

        if n_time < 3:
            issues.append(
                f"Only {n_time} time step(s) — need ≥ 3 for time-derivative equations"
            )
        elif n_time < 30:
            diag_warns.append(
                f"Only {n_time} time steps — seasonal statistics may be unreliable"
            )

        if n_depth < 10:
            issues.append(f"Only {n_depth} depth levels — insufficient for vertical structure")

        # ── 4. Coordinate ranges ─────────────────────────────────────────────
        lon_min   = float(ds.longitude.min())
        lon_max   = float(ds.longitude.max())
        lat_min   = float(ds.latitude.min())
        lat_max   = float(ds.latitude.max())
        depth_min = float(ds.depth.min())
        depth_max = float(ds.depth.max())

        if depth_max < 500:
            diag_warns.append(
                f"Depth only reaches {depth_max:.0f} m — AAIW layer (600–1200 m) not covered"
            )

        print(f"    Grid    : {n_time} × {n_depth} depth × {ds.sizes.get('latitude',0)} lat"
              f" × {ds.sizes.get('longitude',0)} lon")
        print(f"    Domain  : lon [{lon_min:.1f}, {lon_max:.1f}]  "
              f"lat [{lat_min:.1f}, {lat_max:.1f}]  "
              f"depth [{depth_min:.0f}–{depth_max:.0f} m]")

        # ── 5. Temporal continuity ───────────────────────────────────────────
        time_vals = ds.time.values
        t_start   = str(pd.Timestamp(time_vals[0]).date())
        t_end     = str(pd.Timestamp(time_vals[-1]).date())
        print(f"    Period  : {t_start} → {t_end}")

        if n_time > 1:
            dt_days   = np.diff(time_vals) / np.timedelta64(1, "D")
            dt_median = float(np.median(dt_days))
            max_gap   = float(np.max(dt_days))
            if max_gap > dt_median * 2 + 1:
                diag_warns.append(
                    f"Temporal gap: max gap = {max_gap:.1f} days (median = {dt_median:.1f} d)"
                )

        # ── 6. Land-mask baseline NaN fraction (from 2-D zos field) ─────────
        # Used as the expected NaN fraction for the domain (purely land/mask).
        # 3-D variables will have slightly more NaN due to bathymetry — that's normal.
        baseline_nan = 0.0
        if "zos" in ds.data_vars:
            zos_snap = ds["zos"].isel(time=0).compute().values.ravel()
            baseline_nan = float(np.isnan(zos_snap).mean())
        print(f"    Land-mask NaN baseline (from zos): {100*baseline_nan:.1f}%")

        # ── 7. Per-variable checks ───────────────────────────────────────────
        var_stats: dict = {}
        print(f"\n    {'Variable':8s}  {'NaN%':>6}  {'excess':>6}  {'mean':>9}  "
              f"{'std':>9}  {'p5':>9}  {'p95':>9}  check")
        print(f"    {'─'*8}  {'─'*6}  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*6}")

        for var in _REQUIRED_VARS:
            if var not in ds.data_vars:
                var_stats[var] = {}
                continue

            da = ds[var]
            snapshot = da.isel(time=0).compute().values.ravel()
            nan_frac  = float(np.isnan(snapshot).mean())
            nan_excess = nan_frac - baseline_nan  # above land-mask expectation
            finite    = snapshot[np.isfinite(snapshot)]

            if len(finite) == 0:
                issues.append(f"{var}: all values are NaN at t=0")
                var_stats[var] = dict(nan_frac=1.0, mean=np.nan, std=np.nan,
                                      p5=np.nan, p95=np.nan)
                print(f"    {var:8s}  {'100.0':>6}  {'---':>6}  {'ALL NaN':>9}")
                continue

            vmean = float(np.mean(finite))
            vstd  = float(np.std(finite))
            p5    = float(np.percentile(finite, 5))
            p95   = float(np.percentile(finite, 95))
            vmin  = float(np.min(finite))
            vmax  = float(np.max(finite))

            var_stats[var] = dict(nan_frac=nan_frac, mean=vmean, std=vstd, p5=p5, p95=p95)

            check = "OK"
            # Physical range check
            if var in _PHYS_RANGES:
                lo, hi, unit = _PHYS_RANGES[var]
                if vmin < lo - 0.5 * abs(lo) or vmax > hi + 0.5 * abs(hi):
                    issues.append(
                        f"{var}: values [{vmin:.2f}, {vmax:.2f}] {unit} "
                        f"wildly outside plausible range [{lo}, {hi}]"
                    )
                    check = "FAIL"
                elif p5 < lo or p95 > hi:
                    diag_warns.append(
                        f"{var}: p5–p95 = [{p5:.2f}, {p95:.2f}] {unit} "
                        f"partly outside expected [{lo}, {hi}]"
                    )
                    check = "WARN"

            # NaN check: flag only if excess above land-mask baseline is large
            if nan_frac > 0.70:
                issues.append(
                    f"{var}: {100*nan_frac:.1f}% NaN — likely corrupt "
                    f"(land baseline {100*baseline_nan:.1f}%)"
                )
                check = "FAIL"
            elif nan_excess > 0.20:
                diag_warns.append(
                    f"{var}: {100*nan_frac:.1f}% NaN "
                    f"({100*nan_excess:.1f}% above land-mask baseline) — check bathymetry mask"
                )
                check = "WARN"

            print(f"    {var:8s}  {100*nan_frac:6.1f}  {100*nan_excess:+6.1f}  "
                  f"{vmean:9.3g}  {vstd:9.3g}  {p5:9.3g}  {p95:9.3g}  {check}")

        # ── 8. TEOS-10 sanity (SA, CT computable?) ──────────────────────────
        if {"thetao", "so"} <= set(ds.data_vars):
            try:
                import warnings as _warnings  # use alias to avoid name conflict
                depth0  = float(ds.depth.values[0])
                lat_med = float(np.median(ds.latitude.values))
                p0      = gsw.p_from_z(-depth0, lat_med)
                sp_sfc  = ds["so"].isel(time=0, depth=0).compute().values
                pt_sfc  = ds["thetao"].isel(time=0, depth=0).compute().values
                lat2d   = ds.latitude.values[:, None] * np.ones_like(sp_sfc)
                lon2d   = ds.longitude.values[None, :] * np.ones_like(sp_sfc)
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    SA_test = gsw.SA_from_SP(sp_sfc, p0, lon2d, lat2d)
                    CT_test = gsw.CT_from_pt(SA_test, pt_sfc)
                ocean_mask = ~np.isnan(sp_sfc)
                frac_fail  = float(np.isnan(SA_test[ocean_mask]).mean())
                if frac_fail > 0.05:
                    diag_warns.append(
                        f"TEOS-10 SA_from_SP failed for {100*frac_fail:.1f}% of ocean points"
                    )
                else:
                    print(f"\n    TEOS-10 SA/CT: computable (surface, day 0)  "
                          f"SA mean={float(np.nanmean(SA_test)):.3f} g/kg  "
                          f"CT mean={float(np.nanmean(CT_test)):.3f} °C")
            except Exception as exc:
                diag_warns.append(f"TEOS-10 sanity check failed: {exc}")

        # ── 9. Verdict ───────────────────────────────────────────────────────
        if issues:
            verdict = "INVALID"
        elif diag_warns:
            verdict = "VALID_WITH_WARNINGS"
        else:
            verdict = "VALID"

        print(f"\n    Verdict : {verdict}")
        for iss in issues:
            print(f"    ISSUE   : {iss}")
        for w in diag_warns:
            print(f"    WARNING : {w}")

        # ── 10. Figure ────────────────────────────────────────────────────────
        try:
            self._plot_dataset_overview(ds, var_stats, verdict, issues, diag_warns)
        except Exception as exc:
            print(f"    (figure skipped — {exc})")

        notes = "; ".join(issues + diag_warns) if (issues or diag_warns) else "All checks passed."
        return DatasetResult(
            verdict=verdict,
            issues=issues,
            warnings=diag_warns,
            n_time=n_time,
            n_depth=n_depth,
            time_range=(t_start, t_end),
            lon_range=(lon_min, lon_max),
            lat_range=(lat_min, lat_max),
            depth_range=(depth_min, depth_max),
            variable_stats=var_stats,
            notes=notes,
        )

    def _plot_dataset_overview(
        self,
        ds: xr.Dataset,
        var_stats: dict,
        verdict: str,
        issues: list[str],
        warnings: list[str],
    ) -> None:
        """Six-panel overview figure: surface maps + profiles + time series."""
        proj = ccrs.PlateCarree()
        fig  = plt.figure(figsize=(20, 10), constrained_layout=True)

        ax_sst  = fig.add_subplot(2, 3, 1, projection=proj)
        ax_sss  = fig.add_subplot(2, 3, 2, projection=proj)
        ax_spd  = fig.add_subplot(2, 3, 3, projection=proj)
        ax_ssh  = fig.add_subplot(2, 3, 4, projection=proj)
        ax_prof = fig.add_subplot(2, 3, 5)
        ax_mld  = fig.add_subplot(2, 3, 6)

        land_kw = dict(facecolor="#d0d0d0", zorder=3)
        coast_kw = dict(linewidth=0.5, zorder=4)

        def _decorate(ax):
            ax.add_feature(cfeature.LAND, **land_kw)
            ax.add_feature(cfeature.COASTLINE, **coast_kw)

        depth_v = ds.depth.values

        # ── SST ─────────────────────────────────────────────────────────────
        if "thetao" in ds.data_vars:
            sst = ds["thetao"].isel(depth=0).mean("time").compute()
            sst.plot(ax=ax_sst, cmap="RdYlBu_r", transform=proj,
                     cbar_kwargs={"label": "°C", "shrink": 0.8})
            _decorate(ax_sst)
            ax_sst.set_title("Time-mean SST (°C)")

        # ── SSS ─────────────────────────────────────────────────────────────
        if "so" in ds.data_vars:
            sss = ds["so"].isel(depth=0).mean("time").compute()
            sss.plot(ax=ax_sss, cmap="viridis", transform=proj,
                     cbar_kwargs={"label": "PSU", "shrink": 0.8})
            _decorate(ax_sss)
            ax_sss.set_title("Time-mean SSS (PSU)")

        # ── Surface current speed ────────────────────────────────────────────
        if {"uo", "vo"} <= set(ds.data_vars):
            spd = np.sqrt(
                ds["uo"].isel(depth=0).mean("time") ** 2
                + ds["vo"].isel(depth=0).mean("time") ** 2
            ).compute()
            spd.plot(ax=ax_spd, cmap="plasma",
                     vmax=float(np.nanpercentile(spd.values, 98)),
                     transform=proj,
                     cbar_kwargs={"label": "m/s", "shrink": 0.8})
            _decorate(ax_spd)
            ax_spd.set_title("Time-mean surface current speed (m/s)")

        # ── SSH ──────────────────────────────────────────────────────────────
        if "zos" in ds.data_vars:
            ssh = ds["zos"].mean("time").compute()
            vmax_ssh = float(np.nanpercentile(np.abs(ssh.values), 97))
            ssh.plot(ax=ax_ssh, cmap="RdBu_r", vmin=-vmax_ssh, vmax=vmax_ssh,
                     transform=proj,
                     cbar_kwargs={"label": "m", "shrink": 0.8})
            _decorate(ax_ssh)
            ax_ssh.set_title("Time-mean SSH (m)")

        # ── Vertical profiles ────────────────────────────────────────────────
        if "thetao" in ds.data_vars:
            t_prof = ds["thetao"].mean(["time", "latitude", "longitude"]).compute().values
            ax_prof.plot(t_prof, depth_v, "r-", lw=2, label="T (°C)")
            ax_prof.set_xlabel("°C  /  PSU")
        if "so" in ds.data_vars:
            s_prof = ds["so"].mean(["time", "latitude", "longitude"]).compute().values
            ax_prof_s = ax_prof.twiny()
            ax_prof_s.plot(s_prof, depth_v, "b-", lw=2, label="S (PSU)")
            ax_prof_s.set_xlabel("Salinity (PSU)", color="blue")
            ax_prof_s.tick_params(axis="x", colors="blue")
        ax_prof.set_ylim([depth_v[-1] + 50, 0])
        ax_prof.set_ylabel("Depth (m)")
        ax_prof.set_title("Domain-mean T / S profiles")
        ax_prof.grid(alpha=0.3)
        handles = [plt.Line2D([0], [0], color="r", lw=2, label="T (°C)"),
                   plt.Line2D([0], [0], color="b", lw=2, label="S (PSU)")]
        ax_prof.legend(handles=handles, fontsize=8, loc="lower right")

        # ── MLD time series ──────────────────────────────────────────────────
        if "mlotst" in ds.data_vars:
            mld_ts  = ds["mlotst"].mean(["latitude", "longitude"]).compute().values
            t_days  = (ds.time.values - ds.time.values[0]) / np.timedelta64(1, "D")
            ax_mld.plot(t_days, mld_ts, "g-", lw=2)
            ax_mld.set_xlabel("Day")
            ax_mld.set_ylabel("MLD (m)")
            ax_mld.set_title(
                f"Domain-mean MLD  (mean={float(np.nanmean(mld_ts)):.0f} m)"
            )
            ax_mld.grid(alpha=0.3)

        # ── Per-variable NaN summary in MLD panel ────────────────────────────
        if var_stats:
            lines = ["NaN fraction per variable:"]
            for var, st in var_stats.items():
                nf = st.get("nan_frac", np.nan)
                tag = "" if np.isnan(nf) else f"{100*nf:.1f}%"
                lines.append(f"  {var:8s}: {tag}")
            ax_mld.text(
                0.98, 0.97, "\n".join(lines),
                transform=ax_mld.transAxes,
                fontsize=7, va="top", ha="right",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.8),
            )

        # ── Title with verdict ────────────────────────────────────────────────
        colour = {"VALID": "green", "VALID_WITH_WARNINGS": "darkorange",
                  "INVALID": "crimson"}.get(verdict, "black")
        n_issues = len(issues)
        n_warns  = len(warnings)
        fig.suptitle(
            f"Dataset overview — verdict: {verdict}  "
            f"({n_issues} issue(s), {n_warns} warning(s))",
            fontsize=14, fontweight="bold", color=colour,
        )

        _savefig(fig, self.out / "dataset_overview.png")

    # ═══════════════════════════════════════════════════════════════════════
    # EQ-1  TEOS-10 equation of state
    # ═══════════════════════════════════════════════════════════════════════

    def diagnose_eq1_eos(self) -> EqResult:
        """EQ-1: TEOS-10 Equation of State.

        Three residuals quantify EOS-related errors that the PINN will inherit:
          Gap 1 — TEOS-10 vs EOS-80: systematic bias from switching equation of state.
          Gap 2 — TEOS-10 vs linearised EOS: nonlinearity error introduced by the
                  linear EOS often used in PINN loss terms.
          Gap 3 — linearisation error propagated to thermal wind shear (∂u/∂z).

        Each gap is computed twice — once with gsw (NumPy, ground truth) and once
        with gsw_torch (PyTorch, the library that will be used inside the PINN).
        Comparing the two confirms that the PINN minimises the same residuals that
        were measured here.
        """
        print("\n[EQ-1]  TEOS-10 EOS …")
        ds  = self._load()
        drv = self._derived_fields()

        # Retrieve pre-computed TEOS-10 fields (SA, CT, ρ) and grid coordinates.
        # These were computed lazily by _derived_fields() and are backed by Dask.
        SA      = drv["SA"]       # Absolute Salinity       (time, depth, lat, lon)  g/kg
        CT      = drv["CT"]       # Conservative Temperature (time, depth, lat, lon)  °C
        rho_teos = drv["rho"]     # in-situ density via TEOS-10                       kg/m³
        p4      = drv["p"]        # sea pressure (depth, lat)                          dbar
        f_da    = drv["f"]        # Coriolis parameter (lat,)                          s⁻¹

        # Raw GLORYS fields used as input to both EOS libraries.
        sp = ds["so"]       # Practical Salinity  (PSS-78, dimensionless)
        pt = ds["thetao"]   # Potential Temperature (°C, relative to surface)

        # ══════════════════════════════════════════════════════════════════════
        # PART A — Residuals via gsw  (NumPy / C extension, ground truth)
        # Computed over the full time series using lazy xarray / Dask evaluation.
        # ══════════════════════════════════════════════════════════════════════

        # ── Gap 1 (gsw): TEOS-10 rho vs EOS-80 rho ───────────────────────────
        # gsw.rho_t_exact uses potential temperature (pt) directly, as EOS-80 does,
        # whereas TEOS-10 requires Conservative Temperature (CT).  The difference
        # reveals the systematic bias introduced when switching EOS formulations.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho_eos80 = xr.apply_ufunc(
                gsw.rho_t_exact,
                SA, pt, p4,
                dask="parallelized", output_dtypes=[float],
            )

        gap1_gsw = (rho_teos - rho_eos80).compute().values   # (nt, nz, nlat, nlon)
        s1_gsw   = _residual_stats(gap1_gsw)
        print(f"    Gap1 gsw  (TEOS10 − EOS80):   "
              f"mean={s1_gsw['mean']:.4f}  std={s1_gsw['std']:.4f} kg/m³")

        # ── Gap 2 (gsw): TEOS-10 rho vs linearised EOS ───────────────────────
        # The PINN often uses a linear EOS to keep the loss differentiable:
        #   rho_lin = rho0 * (1 - alpha*(CT-CT0) + beta*(SA-SA0))
        # where CT0, SA0 are the domain-and-time-mean reference state at each depth,
        # and alpha, beta are the thermal expansion / haline contraction coefficients
        # evaluated at that reference state.
        # The gap measures the nonlinearity error: how wrong rho_lin is relative to
        # the full TEOS-10 rho.  A large gap means the linear EOS loss is inaccurate.
        CT_mean  = CT.mean(["time", "longitude", "latitude"]).compute()   # (nz,)
        SA_mean  = SA.mean(["time", "longitude", "latitude"]).compute()   # (nz,)
        rho_mean = rho_teos.mean(["time", "longitude", "latitude"]).compute()  # (nz,)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alpha = xr.apply_ufunc(
                gsw.alpha, SA_mean, CT_mean, p4.mean("latitude"),
                dask="parallelized", output_dtypes=[float],
            )
            beta = xr.apply_ufunc(
                gsw.beta,  SA_mean, CT_mean, p4.mean("latitude"),
                dask="parallelized", output_dtypes=[float],
            )

        # Keep (time, depth, lat, lon) order: put 4-D arrays on the left so that
        # xarray uses their dimension order, not the 1-D (depth,) alpha/beta.
        lin_term  = 1.0 - (CT - CT_mean) * alpha + (SA - SA_mean) * beta
        rho_lin   = lin_term * rho_mean
        gap2_gsw  = (rho_teos - rho_lin).compute().values   # (nt, nz, nlat, nlon)
        s2_gsw    = _residual_stats(gap2_gsw)
        print(f"    Gap2 gsw  (TEOS10 − linear):  "
              f"mean={s2_gsw['mean']:.4f}  std={s2_gsw['std']:.4f} kg/m³")

        # ── Gap 3 (gsw): linearisation error → thermal wind shear error ───────
        # Thermal wind: f * ∂u/∂z = -(g/ρ₀) * ∂ρ/∂y
        # If we use rho_lin instead of rho_teos, the meridional density gradient
        # is wrong by (∂rho_lin/∂y - ∂rho_teos/∂y), which implies a spurious
        # vertical shear:  delta(∂u/∂z) = g/(f*ρ₀) * delta(∂ρ/∂y)
        # This is the direct error injected into the thermal wind loss term.
        drho_teos_dy  = _d_dy(rho_teos)
        drho_lin_dy   = _d_dy(rho_lin)
        delta_drho_dy = (drho_lin_dy - drho_teos_dy).compute()

        # Avoid division by zero at the equator (f → 0).
        f_safe    = f_da.where(np.abs(f_da) > 1e-8, other=np.nan)
        delta_dudz_gsw = delta_drho_dy * (G / (f_safe * RHO0))   # s⁻¹/m
        s3_gsw    = _residual_stats(delta_dudz_gsw.values)
        print(f"    Gap3 gsw  (thermal wind Δ∂u/∂z): std={s3_gsw['std']:.2e} s⁻¹/m")

        # ══════════════════════════════════════════════════════════════════════
        # PART B — Residuals via gsw_torch  (PyTorch, used inside the PINN)
        # Computed on a single representative time snapshot (t=0, all depths).
        # Comparing these with Part A confirms that the PINN loss uses the same
        # physics as the diagnostic, to within numerical precision.
        # ══════════════════════════════════════════════════════════════════════

        gsw_torch_notes = "gsw_torch not installed."
        s1_t = s2_t = s3_t = None
        alpha_prof = beta_prof = None
        t_gsw = t_torch = speedup = None
        try:
            import torch
            import gsw_torch as gsw_t
            import time as _time

            depth_v = ds.depth.values          # (nz,)
            lat_v   = ds.latitude.values       # (nlat,)
            lon_v   = ds.longitude.values      # (nlon,)
            p_2d    = drv["p"].values          # (nz, nlat)

            # Build dense 3-D NumPy arrays for the first time snapshot.
            # These are the inputs that the PINN will process at training time.
            sp_np  = ds["so"].isel(time=0).compute().values        # (nz, nlat, nlon)
            pt_np  = ds["thetao"].isel(time=0).compute().values    # (nz, nlat, nlon)
            p_np   = p_2d[:, :, None] * np.ones(sp_np.shape[2])   # (nz, nlat, nlon)
            lon_np = lon_v[None, None, :] * np.ones_like(sp_np)   # (nz, nlat, nlon)
            lat_np = lat_v[None, :, None] * np.ones_like(sp_np)   # (nz, nlat, nlon)

            # gsw ground-truth rho for this snapshot (float64 reference).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                SA_np     = gsw.SA_from_SP(sp_np, p_np, lon_np, lat_np)
                CT_np     = gsw.CT_from_pt(SA_np, pt_np)
                rho_np    = gsw.rho(SA_np, CT_np, p_np)
                rho80_np  = gsw.rho_t_exact(SA_np, pt_np, p_np)

            # Convert to float64 PyTorch tensors.  The PINN will use float32 in
            # production, but we validate in float64 first to isolate algorithmic
            # differences from floating-point rounding.
            def _t64(arr):
                return torch.tensor(arr, dtype=torch.float64)

            SA_t64  = gsw_t.SA_from_SP(_t64(sp_np), _t64(p_np), _t64(lon_np), _t64(lat_np))
            CT_t64  = gsw_t.CT_from_pt(SA_t64, _t64(pt_np))
            p_t64   = _t64(p_np)

            # ── Gap 1 (gsw_torch): TEOS-10 rho vs EOS-80 rho ─────────────────
            # Same comparison as Part A, but using gsw_torch tensors.
            # gsw_torch.rho_t_exact mirrors gsw.rho_t_exact.
            with torch.no_grad():
                rho_teos_t  = gsw_t.rho(SA_t64, CT_t64, p_t64)
                rho_eos80_t = gsw_t.rho_t_exact(SA_t64, _t64(pt_np), p_t64)

            gap1_t = (rho_teos_t - rho_eos80_t).numpy()
            s1_t   = _residual_stats(gap1_t[np.isfinite(gap1_t)])

            # Discrepancy between gsw and gsw_torch for Gap 1.
            disc1 = gap1_t - gap1_gsw[0]   # compare against t=0 slice
            s_disc1 = _residual_stats(disc1[np.isfinite(disc1)])
            print(f"    Gap1 torch (TEOS10 − EOS80):   "
                  f"mean={s1_t['mean']:.4f}  std={s1_t['std']:.4f} kg/m³  "
                  f"|gsw−torch| RMSE={s_disc1['rmse']:.1e}")

            # ── Gap 2 (gsw_torch): TEOS-10 rho vs linearised EOS ─────────────
            # Compute alpha and beta from gsw_torch at the same reference state
            # used in Part A (domain-mean CT0, SA0 at each depth).
            # These calls return tensors, keeping the computation graph intact
            # so that alpha and beta are differentiable w.r.t. SA and CT.
            CT0_np = CT_mean.values       # (nz,) — depth-only vector
            SA0_np = SA_mean.values       # (nz,)
            p0_np  = p_2d.mean(axis=1)   # (nz,) — latitudinal mean pressure

            alpha_t = gsw_t.alpha(_t64(SA0_np), _t64(CT0_np), _t64(p0_np))  # (nz,)
            beta_t  = gsw_t.beta( _t64(SA0_np), _t64(CT0_np), _t64(p0_np))  # (nz,)
            rho0_t  = _t64(rho_mean.values)   # (nz,) reference density

            # Broadcast the 1-D (nz,) coefficients to (nz, nlat, nlon).
            alpha_3d = alpha_t[:, None, None].expand_as(rho_teos_t)
            beta_3d  = beta_t[ :, None, None].expand_as(rho_teos_t)
            rho0_3d  = rho0_t[ :, None, None].expand_as(rho_teos_t)
            CT0_3d   = _t64(CT0_np)[:, None, None].expand_as(rho_teos_t)
            SA0_3d   = _t64(SA0_np)[:, None, None].expand_as(rho_teos_t)

            rho_lin_t  = rho0_3d * (1.0 - alpha_3d * (CT_t64 - CT0_3d)
                                         + beta_3d  * (SA_t64 - SA0_3d))
            gap2_t = (rho_teos_t - rho_lin_t).detach().numpy()
            s2_t   = _residual_stats(gap2_t[np.isfinite(gap2_t)])

            disc2 = gap2_t - gap2_gsw[0]
            s_disc2 = _residual_stats(disc2[np.isfinite(disc2)])
            print(f"    Gap2 torch (TEOS10 − linear):  "
                  f"mean={s2_t['mean']:.4f}  std={s2_t['std']:.4f} kg/m³  "
                  f"|gsw−torch| RMSE={s_disc2['rmse']:.1e}")

            # ── Gap 3 (gsw_torch): linearisation error → thermal wind shear ───
            # Compute the meridional density gradient (∂ρ/∂y) for both rho_teos
            # and rho_lin using the same finite-difference scheme as Part A.
            # We convert back to xarray so _d_dy() can reuse the coordinate metadata.
            def _to_xda(arr_np, name):
                return xr.DataArray(
                    arr_np,
                    coords={"depth": ds.depth, "latitude": ds.latitude,
                            "longitude": ds.longitude},
                    dims=["depth", "latitude", "longitude"],
                    name=name,
                )

            drho_teos_t_dy = _d_dy(_to_xda(rho_teos_t.numpy(), "rho_teos"))
            drho_lin_t_dy  = _d_dy(_to_xda(rho_lin_t.detach().numpy(), "rho_lin"))
            delta_drho_t_dy = (drho_lin_t_dy - drho_teos_t_dy).values  # (nz, nlat, nlon)

            # Broadcast f to (nz, nlat, nlon) and apply the thermal wind formula.
            f_3d = f_da.values[None, :, None] * np.ones_like(delta_drho_t_dy)
            f_3d_safe = np.where(np.abs(f_3d) > 1e-8, f_3d, np.nan)
            delta_dudz_t = (G / (f_3d_safe * RHO0)) * delta_drho_t_dy   # s⁻¹/m
            s3_t = _residual_stats(delta_dudz_t[np.isfinite(delta_dudz_t)])

            disc3 = delta_dudz_t - delta_dudz_gsw.values[0]   # compare t=0 slice
            s_disc3 = _residual_stats(disc3[np.isfinite(disc3)])
            print(f"    Gap3 torch (thermal wind Δ∂u/∂z): "
                  f"std={s3_t['std']:.2e} s⁻¹/m  "
                  f"|gsw−torch| RMSE={s_disc3['rmse']:.1e}")

            # ── Autograd correctness: ∂ρ/∂SA and ∂ρ/∂CT ─────────────────────
            # The PINN uses backpropagation through gsw_torch to compute how the
            # loss changes with respect to SA and CT.  Here we verify that the
            # autograd gradients match central finite differences through gsw,
            # confirming that the PINN receives correct gradient information.
            SA_s = torch.tensor([[35.0]], dtype=torch.float64, requires_grad=True)
            CT_s = torch.tensor([[20.0]], dtype=torch.float64, requires_grad=True)
            p_s  = torch.tensor([[500.0]], dtype=torch.float64)
            gsw_t.rho(SA_s, CT_s, p_s).backward()
            drho_dSA_auto = float(SA_s.grad)
            drho_dCT_auto = float(CT_s.grad)

            eps = 1e-4
            drho_dSA_fd = (gsw.rho(35+eps, 20., 500.) - gsw.rho(35-eps, 20., 500.)) / (2*eps)
            drho_dCT_fd = (gsw.rho(35., 20+eps, 500.) - gsw.rho(35., 20-eps, 500.)) / (2*eps)
            err_SA = abs(drho_dSA_auto - drho_dSA_fd) / abs(drho_dSA_fd)
            err_CT = abs(drho_dCT_auto - drho_dCT_fd) / abs(drho_dCT_fd)
            print(f"    ∂ρ/∂SA: autograd={drho_dSA_auto:.6f}  fd={drho_dSA_fd:.6f}  "
                  f"rel.err={err_SA:.1e}")
            print(f"    ∂ρ/∂CT: autograd={drho_dCT_auto:.6f}  fd={drho_dCT_fd:.6f}  "
                  f"rel.err={err_CT:.1e}")

            # ── Domain-mean EOS gradient profiles (∂ρ/∂SA, ∂ρ/∂CT vs depth) ──
            # Run one backward pass over the full 3-D snapshot to obtain the
            # Jacobian diagonal (element-wise gradients) for SA and CT.
            # These profiles tell the PINN how strongly density responds to
            # salinity vs temperature at each depth — useful for loss weighting.
            SA_g = _t64(SA_np).requires_grad_(True)
            CT_g = _t64(CT_np).requires_grad_(True)
            gsw_t.rho(SA_g, CT_g, p_t64).sum().backward()
            alpha_prof = np.nanmean(SA_g.grad.numpy(), axis=(1, 2))  # (nz,) ∂ρ/∂SA
            beta_prof  = np.nanmean(CT_g.grad.numpy(), axis=(1, 2))  # (nz,) ∂ρ/∂CT

            # ── Speed: gsw (NumPy) vs gsw_torch (f32, no grad) ───────────────
            # On CPU, gsw's C extension is typically faster for single forward
            # passes.  The advantage of gsw_torch is autograd: the PINN can
            # differentiate through the EOS without finite-difference Jacobians.
            n_rep = 3
            t0 = _time.perf_counter()
            for _ in range(n_rep):
                _a = gsw.SA_from_SP(sp_np, p_np, lon_np, lat_np)
                _b = gsw.CT_from_pt(_a, pt_np)
                gsw.rho(_a, _b, p_np)
            t_gsw = (_time.perf_counter() - t0) / n_rep

            def _t32(arr):
                return torch.tensor(arr, dtype=torch.float32)
            t0 = _time.perf_counter()
            for _ in range(n_rep):
                with torch.no_grad():
                    _a = gsw_t.SA_from_SP(_t32(sp_np), _t32(p_np), _t32(lon_np), _t32(lat_np))
                    _b = gsw_t.CT_from_pt(_a, _t32(pt_np))
                    gsw_t.rho(_a, _b, _t32(p_np))
            t_torch = (_time.perf_counter() - t0) / n_rep
            speedup = t_gsw / t_torch
            print(f"    Speed: gsw={t_gsw*1e3:.0f} ms  "
                  f"gsw_torch(f32,no_grad)={t_torch*1e3:.0f} ms  "
                  f"speedup={speedup:.1f}×")

            gsw_torch_notes = (
                f"Gap1 torch RMSE={s1_t['rmse']:.1e} kg/m³ "
                f"(gsw/torch disc={s_disc1['rmse']:.1e}); "
                f"Gap2 torch RMSE={s2_t['rmse']:.1e} kg/m³ "
                f"(disc={s_disc2['rmse']:.1e}); "
                f"Gap3 torch std={s3_t['std']:.1e} s⁻¹/m "
                f"(disc={s_disc3['rmse']:.1e}); "
                f"∂ρ/∂SA rel.err={err_SA:.1e}, "
                f"∂ρ/∂CT rel.err={err_CT:.1e}; "
                f"speedup={speedup:.1f}×"
            )

        except ImportError:
            print("    gsw_torch not installed — skipping Part B")
        except Exception as _exc:
            print(f"    gsw_torch evaluation failed: {_exc}")
            gsw_torch_notes = f"gsw_torch evaluation failed: {_exc}"

        # ══════════════════════════════════════════════════════════════════════
        # Figures
        # ══════════════════════════════════════════════════════════════════════

        depth_v = ds.depth.values

        # ── Figure 1: density gaps and T-S diagram (gsw, full time series) ────
        # Panel 1: domain-mean depth profiles of Gap1 and Gap2.
        # Panel 2: thermal wind shear error profile (Gap3).
        # Panel 3: T-S diagram coloured by |gap2| to show where nonlinearity matters.
        fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)

        ax = axes[0]
        ax.plot(np.nanmean(gap1_gsw, axis=(0, 2, 3)), depth_v, "b-", lw=2,
                label="TEOS10 − EOS80")
        ax.plot(np.nanmean(gap2_gsw, axis=(0, 2, 3)), depth_v, "r-", lw=2,
                label="TEOS10 − linear")
        ax.set_ylim([2500, 0])
        ax.set_xlabel("Δρ (kg/m³)")
        ax.set_ylabel("Depth (m)")
        ax.set_title("EOS density gaps — gsw (full series)")
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1]
        shear_err = np.nanmean(np.abs(delta_dudz_gsw.values), axis=(0, 2, 3))
        ax.plot(shear_err, depth_v, "g-", lw=2)
        ax.set_ylim([2500, 0])
        ax.set_xlabel("|Δ(∂u/∂z)| (s⁻¹/m)")
        ax.set_title("Thermal wind shear error\nfrom linearised EOS (gsw)")
        ax.grid(alpha=0.3)

        ax = axes[2]
        ct_v = CT.isel(time=0).compute().values.ravel()
        sa_v = SA.isel(time=0).compute().values.ravel()
        g2_v = np.abs(gap2_gsw[0]).ravel()
        mask = np.isfinite(ct_v) & np.isfinite(sa_v) & np.isfinite(g2_v)
        sc = ax.scatter(sa_v[mask][::10], ct_v[mask][::10], c=g2_v[mask][::10],
                        s=1, cmap="plasma", vmax=np.nanpercentile(g2_v, 95))
        plt.colorbar(sc, ax=ax, label="|TEOS10 − linear| (kg/m³)")
        ax.set_xlabel("SA (g/kg)")
        ax.set_ylabel("CT (°C)")
        ax.set_title("T–S diagram coloured by |gap2|")
        ax.grid(alpha=0.3)

        _savefig(fig, self.out / "eq1_eos.png")

        # ── Figure 2: gsw_torch evaluation (only when available) ─────────────
        # Panel 1: gap1 and gap2 profiles for gsw vs gsw_torch (snapshot).
        # Panel 2: EOS gradient profiles ∂ρ/∂SA and ∂ρ/∂CT computed via autograd.
        # Panel 3: forward-pass speed comparison (CPU, no grad).
        if s1_t is not None:
            fig4, axes4 = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)

            # Gap profiles: side-by-side gsw vs gsw_torch (snapshot t=0).
            ax = axes4[0]
            ax.plot(np.nanmean(gap1_gsw[0], axis=(1, 2)), depth_v,
                    "b-", lw=2, label="Gap1 gsw")
            ax.plot(np.nanmean(gap1_t,      axis=(1, 2)), depth_v,
                    "b--", lw=2, label="Gap1 torch")
            ax.plot(np.nanmean(gap2_gsw[0], axis=(1, 2)), depth_v,
                    "r-", lw=2, label="Gap2 gsw")
            ax.plot(np.nanmean(gap2_t,      axis=(1, 2)), depth_v,
                    "r--", lw=2, label="Gap2 torch")
            ax.set_ylim([2500, 0])
            ax.set_xlabel("Δρ (kg/m³)")
            ax.set_ylabel("Depth (m)")
            ax.set_title("Gap residuals: gsw vs gsw_torch\n(snapshot t=0)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            # EOS gradient profiles from a single autograd backward pass.
            ax = axes4[1]
            if alpha_prof is not None and beta_prof is not None:
                ax.plot(alpha_prof, depth_v, "b-", lw=2, label="∂ρ/∂SA (g/kg)⁻¹")
                ax.plot(beta_prof,  depth_v, "r-", lw=2, label="∂ρ/∂CT (°C)⁻¹")
                ax.axvline(0, color="gray", lw=0.8)
                ax.legend(fontsize=9)
            else:
                ax.text(0.5, 0.5, "autograd not available", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
            ax.set_ylim([2500, 0])
            ax.set_xlabel("∂ρ/∂(SA or CT)")
            ax.set_title("Autograd EOS gradients\n(domain mean, t=0)")
            ax.grid(alpha=0.3)

            # Speed bar chart.
            ax = axes4[2]
            if t_gsw is not None and t_torch is not None:
                ax.barh(["gsw (NumPy)", "gsw_torch (f32, no_grad)"],
                        [t_gsw * 1e3, t_torch * 1e3],
                        color=["steelblue", "darkorange"])
                ax.set_title(f"Speed: gsw vs gsw_torch (CPU)\n"
                             f"speedup = {speedup:.1f}×")
            else:
                ax.text(0.5, 0.5, "speed benchmark not available", ha="center",
                        va="center", transform=ax.transAxes, color="gray")
                ax.set_title("Speed: gsw vs gsw_torch (CPU)")
            ax.set_xlabel("Wall time (ms) per forward pass")
            ax.grid(axis="x", alpha=0.3)

            _savefig(fig4, self.out / "eq1_gsw_torch.png")

        # ── Verdict ───────────────────────────────────────────────────────────
        # The verdict is based on Gap 2 (linearised EOS error) because that is
        # the residual that directly enters the PINN loss function.
        rmse = s2_gsw["rmse"]
        v    = _verdict(rmse, (0.05, 0.2, 0.5))
        res  = EqResult(
            eq="EQ-1 TEOS-10 EOS",
            verdict=v,
            mean_residual=s2_gsw["mean"],
            std_residual=s2_gsw["std"],
            rmse=rmse,
            p95=s2_gsw["p95"],
            recommended_weight=0.0,   # filled in run_all()
            notes=(
                f"Gap1 gsw mean={s1_gsw['mean']:.4f} kg/m³ (TEOS10 vs EOS80); "
                f"Gap2 gsw std={s2_gsw['std']:.4f} kg/m³ (TEOS10 vs linear EOS); "
                f"Gap3 gsw std={s3_gsw['std']:.2e} s⁻¹/m (thermal wind shear). "
                f"gsw_torch: {gsw_torch_notes}"
            ),
        )
        self._results.append(res)
        return res

    # ═══════════════════════════════════════════════════════════════════════
    # EQ-2  Hydrostatic balance
    # ═══════════════════════════════════════════════════════════════════════

    def diagnose_eq2_hydrostatic(self) -> EqResult:
        print("\n[EQ-2]  Hydrostatic balance …")
        ds  = self._load()
        drv = self._derived_fields()
        rho = drv["rho"]
        p_ref = drv["p"]  # gsw pressure (dbar, 2D: depth×lat)

        # Reconstruct pressure by integrating rho*g downward from surface.
        # p_reconstruct(z) = integral_0^z rho*g dz   [Pa] → convert to dbar
        # 1 Pa = 1e-4 dbar (since 1 dbar = 1e4 Pa)
        rho_v = rho.mean("time").compute().values   # (nz, nlat, nlon)
        depth_v = ds.depth.values                   # (nz,)
        p_integ = sci_int.cumulative_trapezoid(
            rho_v * G, x=depth_v, axis=0, initial=0.0
        ) * 1e-4  # Pa → dbar  shape: (nz, nlat, nlon)

        # Reference: p_from_z averaged over lon
        p_ref_v = p_ref.values  # (nz, nlat)
        p_ref_3d = p_ref_v[:, :, None] * np.ones(rho_v.shape[2])[None, None, :]

        residual_dbar = p_integ - p_ref_3d  # (nz, nlat, nlon)
        # Convert to depth error: dz ≈ dp / (rho*g) * 1e4
        rho_mid = 0.5 * (rho_v[:-1] + rho_v[1:])  # (nz-1, ...)
        residual_m = np.concatenate(
            [residual_dbar[:1] / (RHO0 * G) * 1e4,
             residual_dbar[1:] / (rho_mid * G) * 1e4],
            axis=0
        )  # depth error in metres

        # Stats on domain-mean depth profile (nz,) — not the full (nz, nlat, nlon) field
        s = _residual_stats(np.nanmean(residual_m, axis=(1, 2)))
        print(f"    Hydrostatic depth error (profile): mean={s['mean']:.3f} m  std={s['std']:.3f} m")

        # ── Figure ────────────────────────────────────────────────────────────
        # Mix regular + GeoAxes: profile on the left, maps on the right.
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(18, 5), constrained_layout=True)
        ax_prof  = fig.add_subplot(1, 4, 1)
        ax_500   = fig.add_subplot(1, 4, 2, projection=proj)
        ax_1000  = fig.add_subplot(1, 4, 3, projection=proj)
        ax_2000  = fig.add_subplot(1, 4, 4, projection=proj)

        # Profile
        ax_prof.plot(np.nanmean(residual_m, axis=(1, 2)), depth_v, "b-", lw=2)
        ax_prof.fill_betweenx(
            depth_v,
            np.nanmean(residual_m, axis=(1, 2)) - np.nanstd(residual_m, axis=(1, 2)),
            np.nanmean(residual_m, axis=(1, 2)) + np.nanstd(residual_m, axis=(1, 2)),
            alpha=0.2, color="blue",
        )
        ax_prof.set_ylim([2500, 0])
        ax_prof.set_xlabel("Depth error (m)")
        ax_prof.set_ylabel("Depth (m)")
        ax_prof.set_title("Hydrostatic residual profile")
        ax_prof.grid(alpha=0.3)

        # Maps at 500, 1000, 2000 m
        target_depths = [500, 1000, 2000]
        for ax, tdepth in zip([ax_500, ax_1000, ax_2000], target_depths):
            iz = int(np.argmin(np.abs(depth_v - tdepth)))
            data2d = residual_m[iz]  # (nlat, nlon)
            da = xr.DataArray(
                data2d,
                coords={"latitude": ds.latitude, "longitude": ds.longitude},
                dims=["latitude", "longitude"],
            )
            vmax = float(np.nanpercentile(np.abs(data2d), 95))
            da.plot(
                ax=ax, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                transform=proj,
                cbar_kwargs={"label": "m", "shrink": 0.8},
            )
            ax.set_title(f"Hydrostatic depth error at {tdepth} m")
            ax.add_feature(cfeature.LAND, facecolor="#d0d0d0", zorder=3)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=4)

        _savefig(fig, self.out / "eq2_hydrostatic.png")

        rmse = s["rmse"]
        v = _verdict(rmse, (1.0, 5.0, 20.0))
        res = EqResult(
            eq="EQ-2 Hydrostatic",
            verdict=v,
            mean_residual=s["mean"],
            std_residual=s["std"],
            rmse=rmse,
            p95=s["p95"],
            recommended_weight=0.0,
            notes=f"Depth error from ρg integral: mean={s['mean']:.3f} m, std={s['std']:.3f} m",
        )
        self._results.append(res)
        return res

    # ═══════════════════════════════════════════════════════════════════════
    # EQ-3  Continuity  ∇·u = 0
    # ═══════════════════════════════════════════════════════════════════════

    def diagnose_eq3_continuity(self) -> EqResult:
        print("\n[EQ-3]  Continuity ∇·u = 0 …")
        ds  = self._load()
        drv = self._derived_fields()

        # w is diagnosed from continuity, so dw/dz balances div_h by construction.
        # We report horizontal divergence |∂u/∂x + ∂v/∂y| as the noise floor.
        div_h = drv["div_h"].compute().values  # (time, depth, lat, lon) s⁻¹

        # Normalise by |f| for a dimensionless measure
        f_v = drv["f"].values  # (nlat,)
        f4  = f_v[None, None, :, None] * np.ones(div_h.shape)
        div_norm = div_h / np.abs(f4)  # Rossby-like number for divergence

        # Stats on domain-mean depth profile (nz,) — avoids time/spatial dilution
        s_dim  = _residual_stats(np.nanmean(np.abs(div_h), axis=(0, 2, 3)))
        s_norm = _residual_stats(np.nanmean(np.abs(div_norm), axis=(0, 2, 3)))
        print(f"    |∂u/∂x + ∂v/∂y| (depth profile): RMSE={s_dim['rmse']:.2e} s⁻¹  "
              f"(normalised by |f|: {s_norm['rmse']:.3f})")

        # Time-mean vertical profile of horizontal divergence
        div_prof = np.nanmean(np.abs(div_h), axis=(0, 2, 3))  # (nz,)
        depth_v  = ds.depth.values

        # Map of time-mean |div_h| at surface
        div_sfc = _nanmean_t(np.abs(div_h[:, 0]))  # (nlat, nlon)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

        ax = axes[0]
        ax.plot(div_prof, depth_v, "b-", lw=2)
        ax.set_ylim([2500, 0])
        ax.set_xlabel("|∂u/∂x + ∂v/∂y| (s⁻¹)")
        ax.set_ylabel("Depth (m)")
        ax.set_title("Mean horizontal divergence profile")
        ax.grid(alpha=0.3)
        ax.set_xscale("log")

        ax = axes[1]
        ax.set_projection = ccrs.PlateCarree()
        fig2, axm, proj = _map_axes(1, 1, figsize=(8, 5))
        da_sfc = xr.DataArray(
            div_sfc,
            coords={"latitude": ds.latitude, "longitude": ds.longitude},
            dims=["latitude", "longitude"],
        )
        da_sfc.plot(
            ax=axm.ravel()[0], cmap="Reds",
            vmax=float(np.nanpercentile(div_sfc, 95)),
            transform=proj,
            cbar_kwargs={"label": "s⁻¹", "shrink": 0.8},
        )
        axm.ravel()[0].set_title("Time-mean |∂u/∂x + ∂v/∂y| at surface")
        _savefig(fig2, self.out / "eq3_continuity_map.png")

        fig.axes[1].axis("off")
        # Replace with profile of vertical divergence from diagnosed w
        dw_da = _d_dz(drv["w"])
        total_div = (drv["div_h"] + dw_da).compute().values
        s_tot = _residual_stats(total_div)
        print(f"    |∇·u| (with diagnosed w): RMSE={s_tot['rmse']:.2e} s⁻¹")

        _savefig(fig, self.out / "eq3_continuity_profile.png")

        rmse = s_dim["rmse"]
        v = _verdict(rmse, (1e-7, 5e-7, 2e-6))
        res = EqResult(
            eq="EQ-3 Continuity",
            verdict=v,
            mean_residual=s_dim["mean"],
            std_residual=s_dim["std"],
            rmse=rmse,
            p95=s_dim["p95"],
            recommended_weight=0.0,
            notes=(
                f"Horizontal divergence RMSE={s_dim['rmse']:.2e} s⁻¹ "
                f"(sets noise floor for continuity loss). "
                f"w diagnosed from ∫ div_h dz."
            ),
        )
        self._results.append(res)
        return res

    # ═══════════════════════════════════════════════════════════════════════
    # EQ-4  Geostrophic balance
    # ═══════════════════════════════════════════════════════════════════════

    def diagnose_eq4_geostrophic(self) -> EqResult:
        print("\n[EQ-4]  Geostrophic balance …")
        ds  = self._load()
        drv = self._derived_fields()
        f_da = drv["f"]  # (nlat,)

        f_safe = f_da.where(np.abs(f_da) > 1e-8, other=np.nan)

        zos = ds["zos"]
        # u_geo = -(g/f) * ∂η/∂y  ;  v_geo = (g/f) * ∂η/∂x
        u_geo = -(G / f_safe) * _d_dy(zos)
        v_geo =  (G / f_safe) * _d_dx(zos)

        # Compare with GLORYS surface velocity
        uo_sfc = ds["uo"].isel(depth=0)
        vo_sfc = ds["vo"].isel(depth=0)

        du = (uo_sfc - u_geo).compute().values
        dv = (vo_sfc - v_geo).compute().values
        speed_err = np.sqrt(du**2 + dv**2)

        s_u = _residual_stats(du)
        s_v = _residual_stats(dv)
        s_sp = _residual_stats(speed_err)
        print(f"    |Δu|: RMSE={s_u['rmse']:.4f} m/s   |Δv|: RMSE={s_v['rmse']:.4f} m/s")

        # Rossby number: Ro = |ζ|/|f|
        zeta = (_d_dx(vo_sfc) - _d_dy(uo_sfc)).compute()
        Ro = np.abs(zeta.values) / np.abs(
            f_da.values[None, :, None] * np.ones(zeta.shape)
        )
        frac_qg = float(np.nanmean(Ro < 0.1))
        print(f"    Domain fraction with Ro < 0.1 (QG valid): {100*frac_qg:.1f}%")

        # Time-mean |Δu|
        du_mean = _nanmean_t(np.abs(du))  # (nlat, nlon)
        Ro_mean = _nanmean_t(Ro)         # (nlat, nlon)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True,
                                 subplot_kw={"projection": ccrs.PlateCarree()})
        for ax in axes:
            ax.add_feature(cfeature.LAND, facecolor="#d0d0d0", zorder=3)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=4)

        kw_map = dict(transform=ccrs.PlateCarree())

        da = xr.DataArray(du_mean,
                          coords={"latitude": ds.latitude, "longitude": ds.longitude},
                          dims=["latitude", "longitude"])
        da.plot(ax=axes[0], cmap="Reds",
                vmax=float(np.nanpercentile(du_mean, 95)), **kw_map,
                cbar_kwargs={"label": "m/s", "shrink": 0.8})
        axes[0].set_title("Time-mean |u − u_geo|")

        da_ro = xr.DataArray(Ro_mean,
                             coords={"latitude": ds.latitude, "longitude": ds.longitude},
                             dims=["latitude", "longitude"])
        da_ro.plot(ax=axes[1], cmap="viridis_r", vmax=0.5, **kw_map,
                   cbar_kwargs={"label": "Ro", "shrink": 0.8})
        axes[1].set_title(f"Time-mean Rossby number\n(QG valid: {100*frac_qg:.0f}% of domain)")

        # Scatter |Δu| vs Ro
        axes[2].axis("off")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.scatter(Ro.ravel()[::50], speed_err.ravel()[::50],
                    s=1, alpha=0.3, color="steelblue")
        ax2.axvline(0.1, color="red", ls="--", lw=1.5, label="Ro = 0.1")
        ax2.set_xlabel("Rossby number |ζ|/|f|")
        ax2.set_ylabel("|u − u_geo| (m/s)")
        ax2.set_title("Geostrophic error vs Rossby number")
        ax2.set_xlim([0, 1.0])
        ax2.legend()
        ax2.grid(alpha=0.3)
        _savefig(fig2, self.out / "eq4_geostrophic_scatter.png")

        _savefig(fig, self.out / "eq4_geostrophic.png")

        rmse = s_u["rmse"]
        v = _verdict(rmse, (0.05, 0.15, 0.4))
        res = EqResult(
            eq="EQ-4 Geostrophic",
            verdict=v,
            mean_residual=s_u["mean"],
            std_residual=s_u["std"],
            rmse=rmse,
            p95=s_u["p95"],
            recommended_weight=0.0,
            notes=(
                f"|Δu| RMSE={s_u['rmse']:.4f} m/s; "
                f"QG-valid fraction (Ro<0.1): {100*frac_qg:.0f}%"
            ),
        )
        self._results.append(res)
        return res

    # ═══════════════════════════════════════════════════════════════════════
    # EQ-5  Thermal wind balance
    # ═══════════════════════════════════════════════════════════════════════

    def diagnose_eq5_thermal_wind(self) -> EqResult:
        print("\n[EQ-5]  Thermal wind balance …")
        ds  = self._load()
        drv = self._derived_fields()
        rho = drv["rho"]
        f_da = drv["f"]
        f_safe = f_da.where(np.abs(f_da) > 1e-8, other=np.nan)

        # Thermal wind: f ∂u/∂z = -(g/ρ₀) ∂ρ/∂y  →  residual R_u = ∂u/∂z + (g/(f·ρ₀)) ∂ρ/∂y
        du_dz = _d_dz(ds["uo"])
        dv_dz = _d_dz(ds["vo"])
        drho_dy = _d_dy(rho)
        drho_dx = _d_dx(rho)

        R_u = (du_dz + (G / (f_safe * RHO0)) * drho_dy).compute()
        R_v = (dv_dz - (G / (f_safe * RHO0)) * drho_dx).compute()

        depth_v = ds.depth.values
        # Mask surface Ekman layer (z < 50 m); compute stats on domain-mean depth profile
        iz_50 = int(np.searchsorted(depth_v, 50.0))
        R_u_prof = np.nanmean(R_u.values[:, iz_50:], axis=(0, 2, 3))
        R_v_prof = np.nanmean(R_v.values[:, iz_50:], axis=(0, 2, 3))
        s_u = _residual_stats(R_u_prof)
        s_v = _residual_stats(R_v_prof)
        print(f"    R_u RMSE (z>{depth_v[iz_50]:.0f}m profile)={s_u['rmse']:.2e} s⁻¹/m   "
              f"R_v RMSE={s_v['rmse']:.2e} s⁻¹/m")

        # Also compute the thermal wind shear magnitude for comparison
        tw_shear = np.abs((G / (RHO0 * np.abs(
            drv["f"].values[None, None, :, None]
        ))) * drho_dy.compute().values)
        actual_shear = np.abs(du_dz.compute().values)

        # Mix regular + GeoAxes: two profile axes on the left, two map axes on the right.
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(20, 6), constrained_layout=True)
        ax0    = fig.add_subplot(1, 4, 1)
        ax1    = fig.add_subplot(1, 4, 2)
        ax_200 = fig.add_subplot(1, 4, 3, projection=proj)
        ax_600 = fig.add_subplot(1, 4, 4, projection=proj)

        # Residual profile
        ru_prof = np.nanmean(R_u.values, axis=(0, 2, 3))
        rv_prof = np.nanmean(R_v.values, axis=(0, 2, 3))
        ax0.plot(ru_prof, depth_v, "b-", lw=2, label="R_u = ∂u/∂z + (g/fρ₀)∂ρ/∂y")
        ax0.plot(rv_prof, depth_v, "r--", lw=2, label="R_v = ∂v/∂z − (g/fρ₀)∂ρ/∂x")
        ax0.set_ylim([2500, 0])
        ax0.set_xlabel("Thermal wind residual (s⁻¹/m)")
        ax0.set_ylabel("Depth (m)")
        ax0.set_title("Thermal wind residual profile")
        ax0.legend(fontsize=8)
        ax0.grid(alpha=0.3)

        # Std residual profile
        ru_std = np.nanstd(R_u.values, axis=(0, 2, 3))
        ax1.plot(ru_std, depth_v, "b-", lw=2)
        ax1.plot(np.nanmean(actual_shear, axis=(0, 2, 3)), depth_v, "k--", lw=2,
                 label="|∂u/∂z| (actual shear)")
        ax1.set_ylim([2500, 0])
        ax1.set_xlabel("Magnitude (s⁻¹/m)")
        ax1.set_title("Residual std vs actual shear")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        # Maps at 200 m and 600 m (AAIW level)
        for ax, tdepth in zip([ax_200, ax_600], [200, 600]):
            iz = int(np.argmin(np.abs(depth_v - tdepth)))
            data2d = _nanmean_t(R_u.values[:, iz])
            da = xr.DataArray(data2d,
                              coords={"latitude": ds.latitude, "longitude": ds.longitude},
                              dims=["latitude", "longitude"])
            vmax = float(np.nanpercentile(np.abs(data2d), 95))
            da.plot(ax=ax, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    transform=proj,
                    cbar_kwargs={"label": "s⁻¹/m", "shrink": 0.8},
                    add_colorbar=True)
            ax.add_feature(cfeature.LAND, facecolor="#d0d0d0", zorder=3)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=4)
            ax.set_title(f"Time-mean R_u at {tdepth} m")

        _savefig(fig, self.out / "eq5_thermal_wind.png")

        rmse = s_u["rmse"]
        # Thresholds calibrated for domain-mean depth-profile RMSE (z > 50 m).
        # Typical Agulhas thermocline shear ~1e-4 s⁻¹/m; profile averaging reduces noise.
        v = _verdict(rmse, (5e-6, 2e-5, 1e-4))
        res = EqResult(
            eq="EQ-5 Thermal Wind",
            verdict=v,
            mean_residual=s_u["mean"],
            std_residual=s_u["std"],
            rmse=rmse,
            p95=s_u["p95"],
            recommended_weight=0.0,
            notes=(
                f"R_u RMSE={s_u['rmse']:.2e} s⁻¹/m (depth profile, z>{depth_v[iz_50]:.0f}m); "
                f"R_v RMSE={s_v['rmse']:.2e} s⁻¹/m. "
                "Most important 3D constraint for T/S reconstruction."
            ),
        )
        self._results.append(res)
        return res

    # ═══════════════════════════════════════════════════════════════════════
    # EQ-6  QG potential vorticity
    # ═══════════════════════════════════════════════════════════════════════

    def diagnose_eq6_qgpv(self) -> EqResult:
        print("\n[EQ-6]  QGPV — Rossby number diagnostics …")
        ds  = self._load()
        drv = self._derived_fields()
        f_da = drv["f"]

        uo_sfc = ds["uo"].isel(depth=0)
        vo_sfc = ds["vo"].isel(depth=0)

        # Relative vorticity ζ = ∂v/∂x − ∂u/∂y
        zeta = (_d_dx(vo_sfc) - _d_dy(uo_sfc)).compute()

        f_vals = f_da.values[None, :, None] * np.ones(zeta.shape)
        Ro = np.abs(zeta.values) / np.abs(f_vals)

        frac_qg    = float(np.nanmean(Ro < 0.1))
        frac_qg_3  = float(np.nanmean(Ro < 0.3))
        print(f"    Fraction Ro < 0.1 (QG valid):    {100*frac_qg:.1f}%")
        print(f"    Fraction Ro < 0.3 (Ertel valid): {100*frac_qg_3:.1f}%")

        s = _residual_stats(Ro)

        Ro_mean = _nanmean_t(Ro)  # (nlat, nlon)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True,
                                 subplot_kw={"projection": ccrs.PlateCarree()})
        for ax in axes:
            ax.add_feature(cfeature.LAND, facecolor="#d0d0d0", zorder=3)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=4)

        da_ro = xr.DataArray(Ro_mean,
                             coords={"latitude": ds.latitude, "longitude": ds.longitude},
                             dims=["latitude", "longitude"])
        da_ro.plot(ax=axes[0], cmap="plasma_r", vmin=0, vmax=0.5,
                   transform=ccrs.PlateCarree(),
                   cbar_kwargs={"label": "Ro = |ζ|/|f|", "shrink": 0.8})
        axes[0].set_title(f"Time-mean Rossby number (surface)\nQG valid (Ro<0.1): {100*frac_qg:.0f}%")

        # QG mask: 1 where QG valid, 0 elsewhere
        da_mask = xr.DataArray(
            (Ro_mean < 0.1).astype(float),
            coords={"latitude": ds.latitude, "longitude": ds.longitude},
            dims=["latitude", "longitude"],
        )
        da_mask.plot(ax=axes[1], cmap="RdYlGn", vmin=0, vmax=1,
                     transform=ccrs.PlateCarree(),
                     cbar_kwargs={"label": "QG valid (Ro<0.1)", "shrink": 0.8})
        axes[1].set_title("QGPV validity mask")

        _savefig(fig, self.out / "eq6_qgpv.png")

        # Verdict based on how much of the domain is QG-valid
        v = "USE_WITH_MASK" if frac_qg > 0.5 else "USE_WITH_CAUTION"
        res = EqResult(
            eq="EQ-6 QGPV",
            verdict=v,
            mean_residual=float(np.nanmean(Ro)),
            std_residual=float(np.nanstd(Ro)),
            rmse=float(np.sqrt(np.nanmean(Ro**2))),
            p95=float(np.nanpercentile(Ro, 95)),
            recommended_weight=0.0,
            notes=(
                f"QG valid (Ro<0.1): {100*frac_qg:.0f}% of domain; "
                f"Ertel valid (Ro<0.3): {100*frac_qg_3:.0f}%. "
                "Use with Rossby mask."
            ),
        )
        self._results.append(res)
        return res

    # ═══════════════════════════════════════════════════════════════════════
    # EQ-7  Brunt–Väisälä N²
    # ═══════════════════════════════════════════════════════════════════════

    def diagnose_eq7_brunt_vaisala(self) -> EqResult:
        print("\n[EQ-7]  Brunt–Väisälä N² …")
        ds  = self._load()
        drv = self._derived_fields()
        SA, CT, p4 = drv["SA"], drv["CT"], drv["p"]

        depth_v = ds.depth.values
        lat_v   = ds.latitude.values

        # Compute N² for one time snapshot to save memory, then check full dataset
        SA_v  = SA.isel(time=0).compute().values    # (nz, nlat, nlon)
        CT_v  = CT.isel(time=0).compute().values
        p_v   = p4.values                            # (nz, nlat)
        lat4  = lat_v[:, None] * np.ones((len(lat_v), SA_v.shape[2]))  # (nlat, nlon)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            N2_vals, p_mid_vals = gsw.Nsquared(
                SA_v, CT_v, p_v[:, :, None], lat4[None, :, :],
            )
        # N2_vals shape: (nz-1, nlat, nlon)
        depth_mid = 0.5 * (depth_v[:-1] + depth_v[1:])

        frac_unstable = float(np.nanmean(N2_vals < 0))
        N2_prof_mean  = np.nanmean(N2_vals, axis=(1, 2))
        iz_max_N2     = int(np.argmax(N2_prof_mean))
        print(f"    Fraction N² < 0 (unstable): {100*frac_unstable:.2f}%")
        print(f"    N² maximum at depth: {depth_mid[iz_max_N2]:.0f} m")

        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(16, 6), constrained_layout=True)
        ax_prof     = fig.add_subplot(1, 3, 1)
        ax_thermo   = fig.add_subplot(1, 3, 2, projection=proj)
        ax_unstable = fig.add_subplot(1, 3, 3, projection=proj)

        ax_prof.plot(N2_prof_mean, depth_mid, "b-", lw=2)
        ax_prof.axvline(0, color="red", lw=1, ls="--")
        ax_prof.axhline(depth_mid[iz_max_N2], color="gray", lw=1, ls="--",
                        label=f"N² max at {depth_mid[iz_max_N2]:.0f} m")
        ax_prof.set_ylim([2500, 0])
        ax_prof.set_xlabel("N² (s⁻²)")
        ax_prof.set_ylabel("Depth (m)")
        ax_prof.set_title("Domain-mean N² profile (day 1)")
        ax_prof.legend(fontsize=9)
        ax_prof.grid(alpha=0.3)

        # Map of N² at thermocline level
        iz_thermo = int(np.argmin(np.abs(depth_mid - depth_mid[iz_max_N2])))
        da_N2 = xr.DataArray(
            N2_vals[iz_thermo],
            coords={"latitude": ds.latitude, "longitude": ds.longitude},
            dims=["latitude", "longitude"],
        )
        da_N2.plot(ax=ax_thermo, cmap="Blues",
                   vmax=float(np.nanpercentile(N2_vals[iz_thermo], 95)),
                   transform=proj,
                   cbar_kwargs={"label": "s⁻²", "shrink": 0.8})
        ax_thermo.add_feature(cfeature.LAND, facecolor="#d0d0d0", zorder=3)
        ax_thermo.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=4)
        ax_thermo.set_title(f"N² at thermocline ({depth_mid[iz_thermo]:.0f} m)")

        # Map of any-depth N² < 0 (gravitational instabilities)
        # axis=0 is depth (nz-1) → result shape is (nlat, nlon) ✓
        da_any_unstable = xr.DataArray(
            (N2_vals < 0).any(axis=0).astype(float),
            coords={"latitude": ds.latitude, "longitude": ds.longitude},
            dims=["latitude", "longitude"],
        )
        da_any_unstable.plot(ax=ax_unstable, cmap="Reds",
                             transform=proj,
                             cbar_kwargs={"label": "1 = any N²<0", "shrink": 0.8})
        ax_unstable.add_feature(cfeature.LAND, facecolor="#d0d0d0", zorder=3)
        ax_unstable.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=4)
        ax_unstable.set_title(f"N² < 0 at any depth\n(frac: {100*frac_unstable:.2f}%)")

        _savefig(fig, self.out / "eq7_brunt_vaisala.png")

        # Stats on domain-mean N² depth profile (nz-1,) — not the full (nz-1, nlat, nlon) field.
        # RMSE uses only negative N² values (stability violation) for the verdict.
        s_prof = _residual_stats(N2_prof_mean)
        rmse = float(np.sqrt(np.nanmean(np.minimum(N2_vals, 0)**2)))
        v = _verdict(rmse, (1e-8, 1e-6, 1e-5))
        res = EqResult(
            eq="EQ-7 Brunt-Vaisala N²",
            verdict=v,
            mean_residual=s_prof["mean"],
            std_residual=s_prof["std"],
            rmse=rmse,
            p95=s_prof["p95"],
            recommended_weight=0.0,
            notes=(
                f"N²<0 (unstable) fraction: {100*frac_unstable:.2f}%. "
                f"Thermocline peak at {depth_mid[iz_max_N2]:.0f} m. "
                "Use ReLU(-N²) penalty."
            ),
        )
        self._results.append(res)
        return res

    # ═══════════════════════════════════════════════════════════════════════
    # EQ-8  Ertel potential vorticity
    # ═══════════════════════════════════════════════════════════════════════

    def diagnose_eq8_ertel_pv(self) -> EqResult:
        print("\n[EQ-8]  Ertel PV …")
        ds  = self._load()
        drv = self._derived_fields()
        rho = drv["rho"]
        f_da = drv["f"]
        w_da = drv["w"]

        # Ertel PV (hydrostatic approximation):
        # Q ≈ (1/ρ) * [(f + ζ) * ∂ρ/∂z + ∂u/∂z * ∂ρ/∂y - ∂v/∂z * ∂ρ/∂x]
        zeta_3d = (_d_dx(ds["vo"]) - _d_dy(ds["uo"]))
        f_4d = f_da  # xarray broadcasts (nlat) → (time, depth, lat, lon)
        drho_dz = _d_dz(rho)
        drho_dy = _d_dy(rho)
        drho_dx = _d_dx(rho)
        du_dz   = _d_dz(ds["uo"])
        dv_dz   = _d_dz(ds["vo"])

        Q_ertel = (
            (f_4d + zeta_3d) * drho_dz
            + du_dz * drho_dy
            - dv_dz * drho_dx
        ) / rho

        # QG approximation: Q_qg ≈ (f/ρ) * ∂ρ/∂z  (leading-order stretching)
        Q_qg = (f_4d / rho) * drho_dz

        # Transpose to canonical (time, depth, latitude, longitude) before extracting
        # numpy arrays — xarray broadcasting can silently reorder dimensions.
        _dims = ("time", "depth", "latitude", "longitude")
        Q_e_vals = Q_ertel.transpose(*_dims).compute().values  # (nt, nz, nlat, nlon)
        Q_q_vals = Q_qg.transpose(*_dims).compute().values
        diff     = Q_e_vals - Q_q_vals
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rel_diff = np.abs(diff) / (np.abs(Q_e_vals) + 1e-30)

        s_diff = _residual_stats(rel_diff)
        print(f"    |Q_ertel − Q_qg| / |Q_ertel| mean={s_diff['mean']:.3f}  std={s_diff['std']:.3f}")

        depth_v = ds.depth.values
        # rel_diff shape: (nt, nz, nlat, nlon) — axis labels 0=time,1=depth,2=lat,3=lon
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(13, 5), constrained_layout=True)
        ax_prof = fig.add_subplot(1, 2, 1)
        ax_map  = fig.add_subplot(1, 2, 2, projection=proj)

        ax_prof.plot(np.nanmean(rel_diff, axis=(0, 2, 3)), depth_v, "b-", lw=2)
        ax_prof.set_ylim([2500, 0])
        ax_prof.set_xlabel("|Q_ertel − Q_qg| / |Q_ertel|")
        ax_prof.set_ylabel("Depth (m)")
        ax_prof.set_title("Relative difference: Ertel vs QGPV")
        ax_prof.axvline(0.1, color="red", ls="--", lw=1.5, label="10% threshold")
        ax_prof.legend()
        ax_prof.grid(alpha=0.3)

        # Map at surface — nanmean over time (axis 0), then pick depth index 0
        da_rel = xr.DataArray(
            _nanmean_t(rel_diff[:, 0]),   # (nlat, nlon)
            coords={"latitude": ds.latitude, "longitude": ds.longitude},
            dims=["latitude", "longitude"],
        )
        da_rel.plot(ax=ax_map, cmap="Reds", vmax=1.0,
                    transform=proj,
                    cbar_kwargs={"label": "fraction", "shrink": 0.8})
        ax_map.add_feature(cfeature.LAND, facecolor="#d0d0d0", zorder=3)
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=4)
        ax_map.set_title("|Q_ertel − Q_qg| / |Q_ertel| at surface")

        _savefig(fig, self.out / "eq8_ertel_pv.png")

        rmse = s_diff["rmse"]
        v = "USE_WITH_MASK" if s_diff["mean"] < 0.3 else "USE_WITH_CAUTION"
        res = EqResult(
            eq="EQ-8 Ertel PV",
            verdict=v,
            mean_residual=s_diff["mean"],
            std_residual=s_diff["std"],
            rmse=rmse,
            p95=s_diff["p95"],
            recommended_weight=0.0,
            notes=(
                f"Mean relative difference from QGPV: {100*s_diff['mean']:.1f}%. "
                "Use Ertel PV where Ro > 0.1."
            ),
        )
        self._results.append(res)
        return res

    # ═══════════════════════════════════════════════════════════════════════
    # EQ-9  Salinity advection–diffusion
    # ═══════════════════════════════════════════════════════════════════════

    def diagnose_eq9_salinity_adv(self) -> EqResult:
        print("\n[EQ-9]  Salinity advection–diffusion …")
        ds  = self._load()
        drv = self._derived_fields()
        SA  = drv["SA"]
        w   = drv["w"]

        if ds.sizes["time"] < 3:
            print("    Warning: fewer than 3 time steps — skipping time derivative.")
            res = EqResult(
                eq="EQ-9 Salinity Adv-Diff",
                verdict="USE_WITH_CAUTION",
                mean_residual=np.nan,
                std_residual=np.nan,
                rmse=np.nan,
                p95=np.nan,
                recommended_weight=0.0,
                notes="Insufficient time steps for time-derivative computation.",
            )
            self._results.append(res)
            return res

        # DS/Dt = κ_S * ∂²SA/∂z²
        # Residual: R_S = ∂SA/∂t + u·∂SA/∂x + v·∂SA/∂y + w·∂SA/∂z - κ_S·∂²SA/∂z²
        dSA_dt = _d_dt(SA)
        dSA_dx = _d_dx(SA)
        dSA_dy = _d_dy(SA)
        dSA_dz = _d_dz(SA)
        d2SA_dz2 = _d_dz(dSA_dz)

        R_S = (
            dSA_dt
            + ds["uo"] * dSA_dx
            + ds["vo"] * dSA_dy
            + w * dSA_dz
            - self.kappa_S * d2SA_dz2
        ).compute().values

        depth_v = ds.depth.values
        R_S_prof = np.nanmean(np.abs(R_S), axis=(0, 2, 3))  # (nz,) full profile for plotting
        # Stats on domain-mean depth profile below mean MLD (avoids surface-forcing noise)
        mld_mean_v = float(ds["mlotst"].mean().compute())
        iz_mld = int(np.searchsorted(depth_v, mld_mean_v))
        s = _residual_stats(R_S_prof[iz_mld:])
        print(f"    |R_S| RMSE (z>{depth_v[iz_mld]:.0f}m profile)={s['rmse']:.2e} g/kg/s")

        # Find depth below which |R_S| < threshold
        threshold = 0.1 * np.nanmax(R_S_prof)
        iz_safe = np.argmax(R_S_prof < threshold) if np.any(R_S_prof < threshold) else len(depth_v) - 1
        safe_depth = float(depth_v[iz_safe])
        print(f"    |R_S| < threshold below {safe_depth:.0f} m")

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

        ax = axes[0]
        ax.plot(R_S_prof, depth_v, "b-", lw=2)
        ax.axhline(safe_depth, color="green", ls="--", lw=1.5,
                   label=f"Safe depth: {safe_depth:.0f} m")
        ax.axhspan(600, 1000, alpha=0.1, color="cyan", label="AAIW layer")
        ax.set_ylim([2500, 0])
        ax.set_xlabel("|R_S| (g/kg/day)", labelpad=2)
        ax.set_title("Salinity conservation residual\nvs depth")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # AAIW-layer emphasis (600-1200 m)
        iz_aaiw = np.where((depth_v >= 600) & (depth_v <= 1200))[0]
        R_S_aaiw = np.nanmean(np.abs(R_S[:, iz_aaiw]), axis=(1, 2, 3))
        ax = axes[1]
        t_vals = ds.time.values[1:-1]  # interior time steps (for derivative)
        t_days = (t_vals - t_vals[0]) / np.timedelta64(1, "D")
        n_interior = len(t_vals)
        # dSA_dt has same time coords as interior after finite diff
        ax.plot(t_days[:len(R_S_aaiw)], R_S_aaiw[:len(t_days)],
                "b-", lw=1.5)
        ax.set_xlabel("Day")
        ax.set_ylabel("|R_S| AAIW layer (g/kg/s)")
        ax.set_title("AAIW salinity residual over time\n(600–1200 m depth mean)")
        ax.grid(alpha=0.3)

        _savefig(fig, self.out / "eq9_salinity_adv.png")

        rmse = s["rmse"]
        v = _verdict(rmse, (1e-8, 1e-7, 5e-7))
        res = EqResult(
            eq="EQ-9 Salinity Adv-Diff",
            verdict=v,
            mean_residual=s["mean"],
            std_residual=s["std"],
            rmse=rmse,
            p95=s["p95"],
            recommended_weight=0.0,
            notes=(
                f"|R_S| RMSE={s['rmse']:.2e} g/kg/s (depth profile, z>{depth_v[iz_mld]:.0f}m). "
                f"Conservative below {safe_depth:.0f} m. "
                "AAIW layer (600–1200 m) is key target."
            ),
        )
        self._results.append(res)
        return res

    # ═══════════════════════════════════════════════════════════════════════
    # EQ-10  Temperature advection–diffusion
    # ═══════════════════════════════════════════════════════════════════════

    def diagnose_eq10_temp_adv(self) -> EqResult:
        print("\n[EQ-10]  Temperature advection–diffusion …")
        ds  = self._load()
        drv = self._derived_fields()
        CT  = drv["CT"]
        w   = drv["w"]

        if ds.sizes["time"] < 3:
            res = EqResult(
                eq="EQ-10 Temp Adv-Diff",
                verdict="USE_WITH_CAUTION",
                mean_residual=np.nan,
                std_residual=np.nan,
                rmse=np.nan,
                p95=np.nan,
                recommended_weight=0.0,
                notes="Insufficient time steps for time-derivative computation.",
            )
            self._results.append(res)
            return res

        dCT_dt  = _d_dt(CT)
        dCT_dx  = _d_dx(CT)
        dCT_dy  = _d_dy(CT)
        dCT_dz  = _d_dz(CT)
        d2CT_dz2 = _d_dz(dCT_dz)

        R_T = (
            dCT_dt
            + ds["uo"] * dCT_dx
            + ds["vo"] * dCT_dy
            + w * dCT_dz
            - self.kappa_T * d2CT_dz2
        ).compute().values

        depth_v = ds.depth.values
        R_T_prof = np.nanmean(np.abs(R_T), axis=(0, 2, 3))  # (nz,) full profile for plotting
        # Stats on domain-mean depth profile below mean MLD (avoids surface-forcing noise)
        mld_mean_v = float(ds["mlotst"].mean().compute())
        iz_mld = int(np.searchsorted(depth_v, mld_mean_v))
        R_S_prof_ref = None

        s_T = _residual_stats(R_T_prof[iz_mld:])
        print(f"    |R_T| RMSE (z>{depth_v[iz_mld]:.0f}m profile)={s_T['rmse']:.2e} °C/s")

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

        ax = axes[0]
        ax.plot(R_T_prof, depth_v, "r-", lw=2, label="Temperature")
        ax.axhspan(0, ds["mlotst"].mean().compute().item(), alpha=0.1, color="orange",
                   label=f"Mean MLD ≈ {ds['mlotst'].mean().compute().item():.0f} m")
        ax.set_ylim([2500, 0])
        ax.set_xlabel("|R_T| (°C/s)")
        ax.set_title("Temperature conservation residual\nvs depth")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Seasonal comparison: first half vs second half of record
        nt = R_T.shape[0]
        first_half = np.nanmean(np.abs(R_T[:nt//2]), axis=(0, 2, 3))
        second_half = np.nanmean(np.abs(R_T[nt//2:]), axis=(0, 2, 3))
        ax = axes[1]
        ax.plot(first_half, depth_v, "r-", lw=2, label="First half")
        ax.plot(second_half, depth_v, "r--", lw=2, label="Second half")
        ax.set_ylim([2500, 0])
        ax.set_xlabel("|R_T| (°C/s)")
        ax.set_title("Seasonal comparison of |R_T|")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        _savefig(fig, self.out / "eq10_temp_adv.png")

        rmse = s_T["rmse"]
        v = _verdict(rmse, (1e-8, 1e-7, 5e-7))
        res = EqResult(
            eq="EQ-10 Temp Adv-Diff",
            verdict=v,
            mean_residual=s_T["mean"],
            std_residual=s_T["std"],
            rmse=rmse,
            p95=s_T["p95"],
            recommended_weight=0.0,
            notes=(
                f"|R_T| RMSE={s_T['rmse']:.2e} °C/s (depth profile, z>{depth_v[iz_mld]:.0f}m). "
                "Apply below MLD to avoid surface-forcing contamination."
            ),
        )
        self._results.append(res)
        return res

    # ═══════════════════════════════════════════════════════════════════════
    # EQ-11  Mixed-layer heat budget
    # ═══════════════════════════════════════════════════════════════════════

    def diagnose_eq11_mixed_layer(self) -> EqResult:
        print("\n[EQ-11]  Mixed-layer heat budget …")
        ds  = self._load()
        drv = self._derived_fields()
        CT  = drv["CT"]

        mld = ds["mlotst"]  # (time, lat, lon) metres
        mld_mean = float(mld.mean().compute())

        # Temporal tendency of CT at the surface layer
        # R_ML = ∂CT/∂t (should equal Q_net/(ρ*cp*MLD) but Q_net is unknown)
        CT_sfc = CT.isel(depth=0)
        dCT_dt_sfc = _d_dt(CT_sfc).compute().values  # (time, lat, lon) °C/s

        # Convert to equivalent Q_net: Q = dT/dt * rho * cp * MLD
        mld_v = mld.compute().values[1:-1]  # match interior time steps
        nt_match = min(dCT_dt_sfc.shape[0], mld_v.shape[0])
        Q_implied = dCT_dt_sfc[:nt_match] * RHO0 * CP_SEA * mld_v[:nt_match]  # W/m²

        s = _residual_stats(Q_implied)
        print(f"    Implied Q_net: mean={s['mean']:.1f} W/m²  std={s['std']:.1f} W/m²")

        depth_v = ds.depth.values
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

        # MLD time series (domain mean)
        mld_ts = mld.mean(["latitude", "longitude"]).compute().values
        t_vals = ds.time.values
        t_days = (t_vals - t_vals[0]) / np.timedelta64(1, "D")
        axes[0].plot(t_days, mld_ts, "b-", lw=2)
        axes[0].set_xlabel("Day")
        axes[0].set_ylabel("MLD (m)")
        axes[0].set_title(f"Domain-mean MLD over time\nmean = {mld_mean:.0f} m")
        axes[0].grid(alpha=0.3)

        # MLD map (time-mean)
        mld_tmean = mld.mean("time").compute()
        ax_map = fig.add_subplot(1, 3, 2, projection=ccrs.PlateCarree())
        mld_tmean.plot(ax=ax_map, cmap="plasma_r",
                       transform=ccrs.PlateCarree(),
                       cbar_kwargs={"label": "m", "shrink": 0.8})
        ax_map.add_feature(cfeature.LAND, facecolor="#d0d0d0", zorder=3)
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=4)
        ax_map.set_title("Time-mean MLD")
        axes[1].axis("off")

        # Implied Q_net time series
        t_inner = t_days[1:-1]
        Q_ts = np.nanmean(Q_implied, axis=(1, 2))
        axes[2].plot(t_inner[:len(Q_ts)], Q_ts[:len(t_inner)], "r-", lw=1.5)
        axes[2].axhline(0, color="gray", lw=0.8)
        axes[2].set_xlabel("Day")
        axes[2].set_ylabel("Implied Q_net (W/m²)")
        axes[2].set_title(
            f"Implied surface heat flux\n(std={s['std']:.0f} W/m² — magnitude of missing forcing)"
        )
        axes[2].grid(alpha=0.3)

        _savefig(fig, self.out / "eq11_mixed_layer.png")

        rmse = s["rmse"]
        # Large Q_net std → this loss needs ERA5 forcing or should be masked
        v = _verdict(rmse, (20.0, 80.0, 200.0))
        res = EqResult(
            eq="EQ-11 Mixed Layer",
            verdict=v,
            mean_residual=s["mean"],
            std_residual=s["std"],
            rmse=rmse,
            p95=s["p95"],
            recommended_weight=0.0,
            notes=(
                f"Implied Q_net std={s['std']:.0f} W/m². "
                f"Mean MLD={mld_mean:.0f} m. "
                "Without ERA5 surface forcing this loss is unreliable. "
                "Use as soft seasonal-cycle constraint only."
            ),
        )
        self._results.append(res)
        return res

    # ═══════════════════════════════════════════════════════════════════════
    # Run all and save summary
    # ═══════════════════════════════════════════════════════════════════════

    def run_all(self) -> list[EqResult]:
        """Validate the dataset, then run all 11 equation diagnostics in order."""
        self._results.clear()

        ds_result = self.diagnose_dataset()
        if ds_result.verdict == "INVALID":
            print("\n⚠  Dataset is INVALID — skipping physics diagnostics.")
            for iss in ds_result.issues:
                print(f"   ISSUE: {iss}")
            return []

        runners = [
            self.diagnose_eq1_eos,
            self.diagnose_eq2_hydrostatic,
            self.diagnose_eq3_continuity,
            self.diagnose_eq4_geostrophic,
            self.diagnose_eq5_thermal_wind,
            self.diagnose_eq6_qgpv,
            self.diagnose_eq7_brunt_vaisala,
            self.diagnose_eq8_ertel_pv,
            self.diagnose_eq9_salinity_adv,
            self.diagnose_eq10_temp_adv,
            self.diagnose_eq11_mixed_layer,
        ]
        for fn in runners:
            try:
                fn()
            except Exception as exc:
                print(f"    ERROR in {fn.__name__}: {exc}")

        # Compute recommended weights: 1/std² normalised to max=1
        stds = np.array([r.std_residual for r in self._results])
        stds = np.where(np.isfinite(stds) & (stds > 0), stds, np.nan)
        weights_raw = np.where(np.isfinite(stds), 1.0 / stds**2, 0.0)
        max_w = np.nanmax(weights_raw) if np.any(weights_raw > 0) else 1.0
        weights_norm = weights_raw / max_w
        for r, w in zip(self._results, weights_norm):
            r.recommended_weight = float(w)

        return self._results

    def save_summary(self) -> pathlib.Path:
        """Save summary CSV and print the summary table."""
        if not self._results:
            raise RuntimeError("Call run_all() before save_summary().")

        rows = []
        for r in self._results:
            rows.append(dict(
                equation=r.eq,
                verdict=r.verdict,
                mean_residual=r.mean_residual,
                std_residual=r.std_residual,
                rmse=r.rmse,
                recommended_loss_weight=r.recommended_weight,
                notes=r.notes,
            ))
        df = pd.DataFrame(rows)
        csv_path = self.out / "summary.csv"
        df.to_csv(csv_path, index=False)

        # ── Pretty print ─────────────────────────────────────────────────────
        print("\n" + "═" * 90)
        print("PHYSICS DIAGNOSTICS SUMMARY")
        print("═" * 90)
        fmt = "{:<28}  {:<18}  {:>10}  {:>10}  {:>8}  {:>6}"
        print(fmt.format("Equation", "Verdict", "RMSE", "Std", "Rec.Weight", ""))
        print("─" * 90)
        for r in self._results:
            rmse_s  = f"{r.rmse:.3e}" if np.isfinite(r.rmse) else "  N/A   "
            std_s   = f"{r.std_residual:.3e}" if np.isfinite(r.std_residual) else "  N/A   "
            wt_s    = f"{r.recommended_weight:.3f}" if np.isfinite(r.recommended_weight) else "  N/A"
            print(fmt.format(r.eq, r.verdict, rmse_s, std_s, wt_s, ""))
        print("═" * 90)
        print(f"\nSummary saved to {csv_path}")
        return csv_path


# ═══════════════════════════════════════════════════════════════════════════════
# DiagnosticsRunner — PASS / WARN / FAIL report per equation
# ═══════════════════════════════════════════════════════════════════════════════

class DiagnosticsRunner:
    """Evaluate each physics equation against GLORYS12V1 and print a structured
    PASS / WARN / FAIL report.  Metrics, thresholds, and output format differ
    from EquationDiagnostics (which targets PINN weight selection).

    Usage
    -----
        runner = DiagnosticsRunner("data/glorys_agulhas.zarr")
        runner.run()                        # all equations
        runner.run(["1a", "1b", "5", "7"]) # selected equations
    """

    _SEP = "═" * 54
    _THN = "─" * 54

    def __init__(self, store: str | pathlib.Path,
                 out_dir: str = "results/diagnostics") -> None:
        self.store = pathlib.Path(store)
        self.out   = pathlib.Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self._tm:  dict | None = None   # time-mean 3-D fields cache
        self._t30: tuple | None = None  # (t0, t1) 30-day window
        self._results: list[dict] = []

    # ── data helpers ──────────────────────────────────────────────────────────

    def _open_raw(self) -> xr.Dataset:
        return xr.open_zarr(str(self.store), group="raw", consolidated=False)

    def _30d_window(self) -> tuple[int, int]:
        if self._t30 is None:
            ds = self._open_raw()
            nt = ds.sizes["time"]
            tc = nt // 2
            t0 = max(0, tc - 15)
            self._t30 = (t0, min(nt, t0 + 30))
        return self._t30

    def _load_tm(self) -> dict:
        """Time-mean 3-D numpy arrays for stationary equations."""
        if self._tm is not None:
            return self._tm
        ds = self._open_raw()
        dv = ds.depth.values
        lv = ds.latitude.values
        lov = ds.longitude.values
        nz, nlat, nlon = len(dv), len(lv), len(lov)
        print("  [load] time-mean raw fields …", flush=True)
        sp  = ds["so"].mean("time").compute().values
        pt  = ds["thetao"].mean("time").compute().values
        uo  = ds["uo"].mean("time").compute().values
        vo  = ds["vo"].mean("time").compute().values
        zos = ds["zos"].mean("time").compute().values
        mld = ds["mlotst"].mean("time").compute().values
        p_2d = gsw.p_from_z(-dv[:, None], lv[None, :])        # (nz, nlat)
        p    = p_2d[:, :, None] * np.ones(nlon)                # (nz, nlat, nlon)
        lon3 = lov[None, None, :] * np.ones((nz, nlat, 1))
        lat3 = lv[None, :, None]  * np.ones((nz, 1, nlon))
        print("  [load] SA, CT, ρ …", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            SA  = gsw.SA_from_SP(sp, p, lon3, lat3)
            CT  = gsw.CT_from_pt(SA, pt)
            rho = gsw.rho(SA, CT, p)
        # diagnose w from continuity
        du_dx = self._dx(uo, lov, lv, 2, 1)
        dv_dy = self._dy(vo, lv, 1)
        wo = -sci_int.cumulative_trapezoid(
            du_dx + dv_dy, x=dv, axis=0, initial=0.0)
        f = gsw.f(lv)[:, None] * np.ones(nlon)                 # (nlat, nlon)
        self._tm = dict(
            sp=sp, pt=pt, SA=SA, CT=CT, rho=rho,
            uo=uo, vo=vo, wo=wo, zos=zos, mld=mld,
            p=p, p_2d=p_2d, depth=dv, lat=lv, lon=lov, f=f,
        )
        return self._tm

    # ── grid / derivative helpers ─────────────────────────────────────────────

    @staticmethod
    def _dy(arr: np.ndarray, lat_v: np.ndarray, latax: int) -> np.ndarray:
        """∂arr/∂y [arr_unit / m].  lat_v in degrees."""
        return np.gradient(arr, lat_v, axis=latax) / (np.pi / 180.0 * R_EARTH)

    @staticmethod
    def _dx(arr: np.ndarray, lon_v: np.ndarray,
            lat_v: np.ndarray, lonax: int, latax: int) -> np.ndarray:
        """∂arr/∂x [arr_unit / m].  lon_v, lat_v in degrees."""
        d = np.gradient(arr, lon_v, axis=lonax)          # per degree_lon
        shape = [1] * arr.ndim
        shape[latax] = len(lat_v)
        cos = np.cos(np.deg2rad(lat_v)).reshape(shape)
        return d / (np.pi / 180.0 * R_EARTH * cos)

    @staticmethod
    def _dz(arr: np.ndarray, depth_v: np.ndarray, zax: int) -> np.ndarray:
        """∂arr/∂z [arr_unit / m].  depth_v positive downward."""
        return np.gradient(arr, depth_v, axis=zax)

    @staticmethod
    def _zmean(arr: np.ndarray, depth_v: np.ndarray,
               lo: float, hi: float, zax: int = 0) -> float:
        """Nanmean of arr over depth layer [lo, hi] m (all spatial axes)."""
        mask = (depth_v >= lo) & (depth_v <= hi)
        if not mask.any():
            return np.nan
        idx = [slice(None)] * arr.ndim
        idx[zax] = mask
        return float(np.nanmean(arr[tuple(idx)]))

    # ── verdict / print / figure helpers ─────────────────────────────────────

    @staticmethod
    def _v3(val: float, pass_t: float, warn_t: float,
            low_good: bool = False) -> str:
        """Return PASS/WARN/FAIL.  low_good=True → higher is better."""
        if low_good:
            return "PASS" if val >= pass_t else ("WARN" if val >= warn_t else "FAIL")
        return "PASS" if val <= pass_t else ("WARN" if val <= warn_t else "FAIL")

    def _block(self, eq_id: str, name: str,
               rows: list[tuple[str, str]],
               verdict: str, pinns_note: str) -> None:
        vc = {"PASS": "\033[92m", "WARN": "\033[93m", "FAIL": "\033[91m"}.get(verdict, "")
        rs = "\033[0m"
        print(f"\n  {self._SEP}")
        print(f"  {eq_id} | {name}")
        print(f"  {self._SEP}")
        for lbl, val in rows:
            print(f"  {lbl:<16}: {val}")
        print(f"  {'Verdict':<16}: {vc}{verdict}{rs}")
        print(f"  {'PINNs note':<16}: {pinns_note}")
        print(f"  {self._THN}")

    def _fig_depth(self, slug: str, depth_v: np.ndarray,
                   profile: np.ndarray,
                   pass_t: float, warn_t: float, units: str) -> None:
        """Save |residual| vs depth profile with PASS/WARN/FAIL shading."""
        profile = np.abs(np.asarray(profile, dtype=float))
        fig, ax = plt.subplots(figsize=(5, 7))
        ax.plot(profile, depth_v, "b-", lw=2)
        xmax = max(warn_t * 3.0, float(np.nanpercentile(profile, 98)) * 1.2, warn_t + 1e-30)
        ax.axvspan(0,      pass_t, alpha=0.15, color="green",  label="PASS")
        ax.axvspan(pass_t, warn_t, alpha=0.15, color="orange", label="WARN")
        ax.axvspan(warn_t, xmax,   alpha=0.10, color="red",    label="FAIL")
        ax.set_xlim(0, xmax)
        ax.set_ylim(float(depth_v.max()), float(depth_v.min()))
        ax.set_xlabel(f"|residual| ({units})")
        ax.set_ylabel("Depth (m)")
        ax.set_title(slug)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        path = self.out / f"{slug}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"    saved → {path.name}")

    # ── EQ-1a: TEOS-10 vs EOS-80 ─────────────────────────────────────────────

    def _eq1a(self) -> dict:
        d  = self._load_tm()
        SA, CT, rho = d["SA"], d["CT"], d["rho"]
        p, pt, dv   = d["p"],  d["pt"], d["depth"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho80 = gsw.rho_t_exact(SA, pt, p)
        gap     = rho - rho80
        profile = np.nanmean(np.abs(gap), axis=(1, 2))       # (nz,)
        measured = float(np.nanmean(np.abs(gap)))
        layers = [
            ("surface 0-50m",     0,   50),
            ("therm 100-300m",  100,  300),
            ("interm 600-1200m",600, 1200),
            ("deep 1200-2000m",1200, 2000),
            ("bottom 2000-2500m",2000, 2500),
        ]
        v = self._v3(measured, 0.01, 0.05)
        rows: list[tuple[str, str]] = [
            ("Metric",       "mean(|Δρ|) across domain"),
            ("Measured",     f"{measured:.4f} kg/m³"),
            ("Acceptable",   "< 0.01 kg/m³  (Argo sensor floor)"),
            ("Noise floor",  "~0.02 kg/m³  (EOS-80 → TEOS-10 bias)"),
        ]
        for lname, lo, hi in layers:
            rows.append((f"  {lname}", f"{self._zmean(np.abs(gap), dv, lo, hi):.4f} kg/m³"))
        note = ("TEOS-10 bias over EOS-80 is systematic; use TEOS-10 everywhere in PINN."
                if measured < 0.05 else
                "Large EOS gap; never use EOS-80 in loss — TEOS-10 only.")
        self._block("EQ-1a", "TEOS-10 EOS (full)", rows, v, note)
        self._fig_depth("eq1a_eos_full", dv, profile, 0.01, 0.05, "kg/m³")
        r = dict(eq="EQ-1a", name="TEOS-10 EOS (full)",
                 measured=measured, acceptable="< 0.01 kg/m³", verdict=v, units="kg/m³")
        self._results.append(r)
        return r

    # ── EQ-1b: linearised EOS ────────────────────────────────────────────────

    def _eq1b(self) -> dict:
        d  = self._load_tm()
        SA, CT, rho = d["SA"], d["CT"], d["rho"]
        p,  uo, dv  = d["p"],  d["uo"], d["depth"]
        lv          = d["lat"]
        f_safe = np.where(np.abs(d["f"]) > 1e-8, d["f"], np.nan)
        # Reference state at each depth level (horizontal mean)
        CT0 = np.nanmean(CT,  axis=(1, 2), keepdims=True)   # (nz, 1, 1)
        SA0 = np.nanmean(SA,  axis=(1, 2), keepdims=True)
        p0  = np.nanmean(p,   axis=(1, 2), keepdims=True)
        CT0e = np.broadcast_to(CT0, CT.shape).copy()
        SA0e = np.broadcast_to(SA0, SA.shape).copy()
        p0e  = np.broadcast_to(p0,  p.shape).copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho0  = gsw.rho(SA0e, CT0e, p0e)
            alpha0 = gsw.alpha(SA0e, CT0e, p0e)
            beta0  = gsw.beta( SA0e, CT0e, p0e)
        rho_lin = rho0 * (1.0 - alpha0 * (CT - CT0) + beta0 * (SA - SA0))
        gap2    = rho - rho_lin
        # Thermocline metric
        therm_gap = self._zmean(np.abs(gap2), dv, 100, 600)
        # Thermal wind error ratio at thermocline
        drho_diff = self._dy(rho - rho_lin, lv, 1)
        du_dz_err = np.abs((G / (f_safe[None] * RHO0)) * drho_diff)
        du_dz_sig = np.abs(self._dz(uo, dv, 0)) + 1e-10
        msk_th = (dv >= 100) & (dv <= 600)
        ratio  = float(np.nanmean(
            (du_dz_err / du_dz_sig)[msk_th]))
        aaiw_gap = self._zmean(np.abs(gap2), dv, 600, 1200)
        if therm_gap < 0.05 and ratio < 0.1:
            v = "PASS"
        elif therm_gap > 0.1 or ratio > 0.2:
            v = "FAIL"
        else:
            v = "WARN"
        profile = np.nanmean(np.abs(gap2), axis=(1, 2))
        rows: list[tuple[str, str]] = [
            ("Metric",        "mean|Δρ_lin|(100-600m) + TW ratio"),
            ("Measured gap2", f"{therm_gap:.4f} kg/m³  (therm 100-600m)"),
            ("Acceptable",    "< 0.05 kg/m³  AND  TW ratio < 0.1"),
            ("TW ratio",      f"{ratio:.4f}"),
            ("AAIW gap2",     f"{aaiw_gap:.4f} kg/m³  (600-1200m)"),
        ]
        if aaiw_gap > 0.05:
            rows.append(("  ⚠ AAIW flag",
                         f"AAIW Δρ_lin={aaiw_gap:.3f} > 0.05 — use full TEOS-10"))
        note = ("Linearised EOS acceptable in thermocline; monitor AAIW layer." if v == "PASS"
                else "Linearised EOS error significant; use full TEOS-10 in density loss.")
        self._block("EQ-1b", "TEOS-10 EOS (linearised)", rows, v, note)
        self._fig_depth("eq1b_eos_linear", dv, profile, 0.05, 0.10, "kg/m³")
        r = dict(eq="EQ-1b", name="TEOS-10 EOS (linearised)",
                 measured=therm_gap, acceptable="< 0.05 kg/m³ (therm)", verdict=v, units="kg/m³")
        self._results.append(r)
        return r

    # ── EQ-2: hydrostatic ────────────────────────────────────────────────────

    def _eq2(self) -> dict:
        d   = self._load_tm()
        rho, dv = d["rho"], d["depth"]
        # cumulative pressure from density integration (Pa)
        rho_mid  = 0.5 * (rho[:-1] + rho[1:])          # (nz-1, nlat, nlon)
        dz_arr   = np.diff(dv)[:, None, None]
        p_int_Pa = np.cumsum(rho_mid * G * dz_arr, axis=0)
        # reference pressure from gsw at depth midpoints (dbar → Pa)
        p_exact  = d["p"][1:] * 1e4                     # (nz-1, nlat, nlon)
        delta_z  = (p_int_Pa - p_exact) / (RHO0 * G)   # m
        mean_dz  = float(np.nanmean(np.abs(delta_z)))
        dv_mid   = 0.5 * (dv[:-1] + dv[1:])
        # values at specific target depths
        check = [500, 1000, 1500, 2000, 2500]
        depth_vals = {
            zc: float(np.nanmean(np.abs(delta_z[np.argmin(np.abs(dv_mid - zc))])))
            for zc in check
        }
        max_2500 = depth_vals[2500]
        if mean_dz < 1.0 and max_2500 < 5.0:    v = "PASS"
        elif mean_dz > 3.0 or max_2500 > 15.0:  v = "FAIL"
        else:                                     v = "WARN"
        profile = np.nanmean(np.abs(delta_z), axis=(1, 2))
        rows: list[tuple[str, str]] = [
            ("Metric",      "mean|Δz| domain + max|Δz| at 2500m"),
            ("Measured",    f"{mean_dz:.3f} m"),
            ("Acceptable",  "< 1m mean  AND  < 5m at 2500m"),
            ("Max at 2500m", f"{max_2500:.2f} m"),
        ]
        for zc in check:
            rows.append((f"  Δz at {zc}m", f"{depth_vals[zc]:.2f} m"))
        note = ("Hydrostatic holds; pressure-gradient loss is numerically stable." if v == "PASS"
                else "Hydrostatic pressure error elevated; check vertical resolution.")
        self._block("EQ-2", "Hydrostatic balance", rows, v, note)
        self._fig_depth("eq2_hydrostatic", dv_mid, profile, 1.0, 3.0, "m")
        r = dict(eq="EQ-2", name="Hydrostatic balance",
                 measured=mean_dz, acceptable="< 1 m", verdict=v, units="m")
        self._results.append(r)
        return r

    # ── EQ-3: continuity ─────────────────────────────────────────────────────

    def _eq3(self) -> dict:
        d  = self._load_tm()
        uo, vo, wo = d["uo"], d["vo"], d["wo"]
        dv, lv, lov = d["depth"], d["lat"], d["lon"]
        div = (self._dx(uo, lov, lv, 2, 1)
             + self._dy(vo, lv, 1)
             + self._dz(wo, dv, 0))
        mean_div  = float(np.nanmean(np.abs(div)))
        f_3d      = d["f"][None] * np.ones_like(div)
        mean_norm = float(np.nanmean(np.abs(div) / (np.abs(f_3d) + 1e-12)))
        frac_hi   = float(np.nanmean((np.abs(div) > 1e-7).astype(float)))
        if mean_div < 1e-7 and mean_norm < 1e-3:    v = "PASS"
        elif mean_div > 5e-7 or mean_norm > 5e-3:   v = "FAIL"
        else:                                         v = "WARN"
        profile = np.nanmean(np.abs(div), axis=(1, 2))
        rows: list[tuple[str, str]] = [
            ("Metric",      "mean|∇·u| and |∇·u|/|f|"),
            ("Measured div", f"{mean_div:.2e} s⁻¹"),
            ("Acceptable",  "< 1e-7 s⁻¹  AND  norm < 1e-3"),
            ("div_norm",    f"{mean_norm:.2e}"),
            ("Frac > 1e-7", f"{frac_hi:.1%}  (above noise floor)"),
        ]
        note = ("Continuity holds; diagnosed w reliable for vertical advection." if v == "PASS"
                else "Elevated divergence; apply continuity loss with reduced weight.")
        self._block("EQ-3", "Continuity", rows, v, note)
        self._fig_depth("eq3_continuity", dv, profile, 1e-7, 5e-7, "s⁻¹")
        r = dict(eq="EQ-3", name="Continuity",
                 measured=mean_div, acceptable="< 1e-7 s⁻¹", verdict=v, units="s⁻¹")
        self._results.append(r)
        return r

    # ── EQ-4: geostrophic ────────────────────────────────────────────────────

    def _eq4(self) -> dict:
        d   = self._load_tm()
        uo, vo, zos = d["uo"], d["vo"], d["zos"]
        lv, lov     = d["lat"], d["lon"]
        f           = d["f"]
        f_safe      = np.where(np.abs(f) > 1e-8, f, np.nan)
        u_geo = -(G / f_safe) * self._dy(zos, lv, 0)
        v_geo =  (G / f_safe) * self._dx(zos, lov, lv, 1, 0)
        delta_u = np.abs(uo[0] - u_geo)
        zeta    = self._dx(vo[0], lov, lv, 1, 0) - self._dy(uo[0], lv, 0)
        Ro      = np.abs(zeta) / (np.abs(f) + 1e-12)
        m_lo, m_hi, m_vhi = Ro < 0.1, Ro > 0.1, Ro > 0.2
        frac_g  = float(np.nanmean(m_lo.astype(float)))
        frac_ag = float(np.nanmean(m_vhi.astype(float)))
        mean_Ro = float(np.nanmean(Ro))
        du_g  = float(np.nanmean(delta_u[m_lo]))  if m_lo.any()  else np.nan
        du_ag = float(np.nanmean(delta_u[m_hi]))  if m_hi.any()  else np.nan
        if du_g < 0.05 and frac_g > 0.60:   v = "PASS"
        elif du_g > 0.15 or frac_g < 0.40:  v = "FAIL"
        else:                                 v = "WARN"
        rows: list[tuple[str, str]] = [
            ("Metric",       "mean|Δu| (Ro<0.1) + frac(Ro<0.1)"),
            ("Measured |Δu|", f"{du_g:.4f} m/s  (Ro<0.1 region)"),
            ("Acceptable",   "< 0.05 m/s  AND  frac(Ro<0.1) > 0.60"),
            ("|Δu| Ro>0.1",  f"{du_ag:.4f} m/s"),
            ("frac(Ro<0.1)", f"{frac_g:.1%}"),
            ("mean Ro",      f"{mean_Ro:.3f}"),
            ("frac(Ro>0.2)", f"{frac_ag:.1%}  (strongly ageostrophic)"),
        ]
        note = ("Geostrophic balance holds in QG region; use SSH constraint with Ro mask."
                if v != "FAIL" else
                "Geostrophic residual too large; use strict Ro<0.1 mask on SSH loss.")
        self._block("EQ-4", "Geostrophic balance", rows, v, note)
        # lat vs delta_u figure
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(np.nanmean(delta_u, axis=1), lv, "b-", lw=2)
        ax.axvline(0.05, color="green", ls="--", label="PASS 0.05 m/s")
        ax.axvline(0.15, color="red",   ls="--", label="FAIL 0.15 m/s")
        ax.set_xlabel("|Δu| (m/s)"); ax.set_ylabel("Latitude (°)")
        ax.set_title("EQ-4 Geostrophic"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        p4path = self.out / "eq4_geostrophic.png"
        fig.savefig(p4path, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"    saved → {p4path.name}")
        r = dict(eq="EQ-4", name="Geostrophic balance",
                 measured=du_g, acceptable="< 0.05 m/s (Ro<0.1)", verdict=v, units="m/s")
        self._results.append(r)
        return r

    # ── EQ-5: thermal wind ───────────────────────────────────────────────────

    def _eq5(self) -> dict:
        d   = self._load_tm()
        uo, rho     = d["uo"], d["rho"]
        dv, lv, lov = d["depth"], d["lat"], d["lon"]
        f_safe = np.where(np.abs(d["f"]) > 1e-8, d["f"], np.nan)
        du_dz_obs = self._dz(uo,  dv, 0)
        drho_dy   = self._dy(rho, lv, 1)
        du_dz_tw  = -(G / (f_safe[None] * RHO0)) * drho_dy
        residual  = du_dz_obs - du_dz_tw
        layers = [
            ("surface 0-50m",      0,   50),
            ("ekman 50-100m",     50,  100),
            ("therm 100-600m",   100,  600),
            ("interm 600-1200m", 600, 1200),
            ("deep 1200-2500m", 1200, 2500),
        ]
        ve_by_layer: dict[str, float] = {}
        for lname, lo, hi in layers:
            mask = (dv >= lo) & (dv <= hi)
            if not mask.any():
                ve_by_layer[lname] = np.nan; continue
            obs_l = du_dz_obs[mask].ravel()
            res_l = residual[mask].ravel()
            fin   = np.isfinite(obs_l) & np.isfinite(res_l)
            if fin.sum() < 10:
                ve_by_layer[lname] = np.nan; continue
            ve_by_layer[lname] = float(
                1.0 - np.var(res_l[fin]) / (np.var(obs_l[fin]) + 1e-30))
        ve_therm = ve_by_layer["therm 100-600m"]
        msk_th   = (dv >= 100) & (dv <= 600)
        obs_t = du_dz_obs[msk_th]; res_t = residual[msk_th]
        fin   = np.isfinite(obs_t) & np.isfinite(res_t)
        ratio = float(np.nanmean(np.abs(res_t[fin])) /
                      (np.nanmean(np.abs(obs_t[fin])) + 1e-12))
        if ve_therm > 0.70 and ratio < 0.30:   v = "PASS"
        elif ve_therm < 0.40 or ratio > 0.50:  v = "FAIL"
        else:                                    v = "WARN"
        profile = np.nanmean(np.abs(residual), axis=(1, 2))
        rows: list[tuple[str, str]] = [
            ("Metric",        "var_explained (therm) + |res|/|obs|"),
            ("var_explained", f"{ve_therm:.3f}  (therm 100-600m)"),
            ("Acceptable",    "var_exp > 0.70  AND  ratio < 0.30"),
            ("|res|/|obs|",   f"{ratio:.3f}  (therm 100-600m)"),
        ]
        for lname, lo, hi in layers:
            rows.append((f"  ve {lname}", f"{ve_by_layer[lname]:.3f}"))
        note = ("Thermal wind reliable in thermocline; primary 3D constraint for T/S." if v == "PASS"
                else "Thermal wind weaker than expected; mask Ekman layer, check Ro.")
        self._block("EQ-5", "Thermal wind", rows, v, note)
        self._fig_depth("eq5_thermal_wind", dv, profile, 1e-6, 5e-6, "s⁻¹/m")
        r = dict(eq="EQ-5", name="Thermal wind",
                 measured=ve_therm, acceptable="> 0.70 var_explained", verdict=v,
                 units="dimensionless")
        self._results.append(r)
        return r

    # ── EQ-6: QGPV (barotropic surface) ──────────────────────────────────────

    def _eq6(self) -> dict:
        ds   = self._open_raw()
        t0, t1 = self._30d_window()
        ds30 = ds.isel(time=slice(t0, t1))
        lv   = ds.latitude.values; lov = ds.longitude.values
        dv   = ds.depth.values
        dt_s = float((ds.time.values[1] - ds.time.values[0])
                     / np.timedelta64(1, "s"))
        print("  [EQ-6] loading surface 30d …", flush=True)
        uo_s = ds30["uo"].isel(depth=0).compute().values  # (n30, nlat, nlon)
        vo_s = ds30["vo"].isel(depth=0).compute().values
        zos  = ds30["zos"].compute().values
        f    = gsw.f(lv)[:, None] * np.ones(len(lov))
        f_safe = np.where(np.abs(f) > 1e-8, f, np.nan)
        zeta = (self._dx(vo_s, lov, lv, 2, 1)
               - self._dy(uo_s, lv, 1))
        q    = zeta + f[None]
        psi  = G * zos / f_safe[None]
        J    = (self._dx(psi, lov, lv, 2, 1) * self._dy(q,   lv, 1)
               - self._dy(psi, lv, 1)         * self._dx(q, lov, lv, 2, 1))
        dq_dt = np.gradient(q, dt_s, axis=0)
        conserv = dq_dt + J
        Ro    = np.abs(zeta) / (np.abs(f[None]) + 1e-12)
        m_g   = Ro < 0.1
        frac_g = float(np.nanmean(m_g.astype(float)))
        qf     = np.abs(q * f[None]) + 1e-20
        ratio  = float(np.nanmean(np.abs(conserv[m_g]) / qf[m_g]))
        mean_zeta = float(np.nanmean(np.abs(zeta)))
        mean_f    = float(np.nanmean(np.abs(f)))
        if ratio < 0.10 and frac_g > 0.60:   v = "PASS"
        elif ratio > 0.20 or frac_g < 0.40:  v = "FAIL"
        else:                                  v = "WARN"
        rows: list[tuple[str, str]] = [
            ("Metric",        "|∂q/∂t + J(ψ,q)| / |qf|  where Ro<0.1"),
            ("Measured ratio", f"{ratio:.4f}"),
            ("Acceptable",    "< 0.10  AND  frac(Ro<0.1) > 0.60"),
            ("frac(Ro<0.1)",  f"{frac_g:.1%}"),
            ("mean|ζ|",       f"{mean_zeta:.2e} s⁻¹"),
            ("mean|f|",       f"{mean_f:.2e} s⁻¹"),
        ]
        note = ("Barotropic QGPV conserved in QG region; use as soft vorticity constraint."
                if v != "FAIL" else
                "QGPV conservation residual too large; use only with Ro<0.1 mask.")
        self._block("EQ-6", "QGPV (barotropic surface)", rows, v, note)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(np.nanmean(np.abs(conserv), axis=(0, 2)), lv, "b-", lw=2)
        ax.set_xlabel("|∂q/∂t + J| (s⁻²)"); ax.set_ylabel("Latitude (°)")
        ax.set_title("EQ-6 QGPV conservation"); ax.grid(alpha=0.3)
        p6 = self.out / "eq6_qgpv.png"
        fig.savefig(p6, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"    saved → {p6.name}")
        r = dict(eq="EQ-6", name="QGPV (barotropic surface)",
                 measured=ratio, acceptable="< 0.10", verdict=v, units="dimensionless")
        self._results.append(r)
        return r

    # ── EQ-7: Brunt-Väisälä N² ───────────────────────────────────────────────

    def _eq7(self) -> dict:
        d   = self._load_tm()
        SA, CT, p_2d = d["SA"], d["CT"], d["p_2d"]
        dv, lv, lov  = d["depth"], d["lat"], d["lon"]
        mld          = d["mld"]
        nlon = len(lov)
        lat_2d = lv[:, None] * np.ones(nlon)
        # p broadcast to (nz, nlat, nlon) → take mean over lon for gsw
        p_2d_nlon = d["p"][:, :, 0]                            # (nz, nlat)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            N2_mid, _ = gsw.Nsquared(SA, CT, p_2d_nlon[:, :, None] * np.ones(nlon),
                                     lat_2d[None, :, :] * np.ones((len(dv), 1, 1)))
        dv_mid = 0.5 * (dv[:-1] + dv[1:])
        mld_3d = mld[None, :, :] * np.ones((len(dv_mid), 1, 1))
        dm_3d  = dv_mid[:, None, None] * np.ones_like(N2_mid)
        below  = dm_3d > mld_3d
        N2_bl  = np.where(below, N2_mid, np.nan)
        frac_unstable = float(np.nanmean((N2_bl < 0).astype(float)))
        n2_therm      = self._zmean(N2_mid, dv_mid, 100, 300)
        if frac_unstable < 0.05 and n2_therm > 1e-5:   v = "PASS"
        elif frac_unstable > 0.15 or n2_therm < 1e-6:  v = "FAIL"
        else:                                             v = "WARN"
        layers = [
            ("surface 0-50m",    0,   50),
            ("therm 100-300m",  100,  300),
            ("interm 600-1200m",600, 1200),
            ("deep 1200-2000m",1200, 2000),
        ]
        profile = np.nanmean(N2_mid, axis=(1, 2))
        rows: list[tuple[str, str]] = [
            ("Metric",       "frac(N²<0 below MLD) + mean N² (therm)"),
            ("frac unstable", f"{frac_unstable:.2%}  (N²<0 below MLD)"),
            ("Acceptable",   "frac < 0.05  AND  N²(therm) > 1e-5 s⁻²"),
            ("N²(therm)",    f"{n2_therm:.2e} s⁻²  (100-300m)"),
            ("MLD min/max",  f"{float(np.nanmin(mld)):.1f} / {float(np.nanmax(mld)):.1f} m"),
        ]
        for lname, lo, hi in layers:
            rows.append((f"  N² {lname}", f"{self._zmean(N2_mid, dv_mid, lo, hi):.2e} s⁻²"))
        note = ("Stratification stable; N² usable directly in QGPV and buoyancy terms." if v == "PASS"
                else "Occasional N²<0 patches below MLD; add ReLU(−N²) penalty to buoyancy loss.")
        self._block("EQ-7", "Brunt-Väisälä N²", rows, v, note)
        self._fig_depth("eq7_brunt_vaisala", dv_mid, profile, 1e-5, 1e-6, "s⁻²")
        r = dict(eq="EQ-7", name="Brunt-Väisälä N²",
                 measured=frac_unstable, acceptable="< 0.05 frac", verdict=v, units="fraction")
        self._results.append(r)
        return r

    # ── EQ-8: Ertel PV ───────────────────────────────────────────────────────

    def _eq8(self) -> dict:
        # Compare full Ertel PV against its QG linearisation in consistent units.
        # Q_ertel = (1/ρ)(f+ζ) ∂ρ/∂z  [m⁻¹ s⁻¹]
        # Q_lin   = -(f+ζ) N²/g        [m⁻¹ s⁻¹]  (using ∂ρ/∂z = -ρN²/g)
        # diff_frac = |Q_ertel - Q_lin| / |Q_ertel|  — dimensionless, meaningful
        d   = self._load_tm()
        SA, CT, rho = d["SA"], d["CT"], d["rho"]
        uo, vo      = d["uo"], d["vo"]
        dv, lv, lov = d["depth"], d["lat"], d["lon"]
        f           = d["f"]
        nlon        = len(lov)
        zeta = (self._dx(vo, lov, lv, 2, 1) - self._dy(uo, lv, 1))  # (nz, nlat, nlon)
        drho_dz = self._dz(rho, dv, 0)
        Q_ertel = (1.0 / (rho + 1e-10)) * (f[None] + zeta) * drho_dz   # m⁻¹ s⁻¹
        # N² from gsw.Nsquared, mid-points (nz-1, nlat, nlon); interpolate to full levels
        lat_3d = lv[:, None] * np.ones(nlon)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            N2_mid, _ = gsw.Nsquared(
                SA, CT,
                d["p"][:, :, 0][:, :, None] * np.ones(nlon),
                lat_3d[None] * np.ones((len(dv), 1, 1)),
            )
        # Interpolate N2 from mid-points to full levels by padding edges
        N2 = np.concatenate([N2_mid[:1], 0.5*(N2_mid[:-1]+N2_mid[1:]), N2_mid[-1:]], axis=0)
        N2 = np.clip(N2, 1e-8, None)                                    # avoid negative
        Q_lin = -(f[None] + zeta) * N2 / G                              # m⁻¹ s⁻¹
        diff_frac = np.abs(Q_ertel - Q_lin) / (np.abs(Q_ertel) + 1e-20)
        Ro        = np.abs(zeta) / (np.abs(f[None]) + 1e-12)
        m_ag      = Ro > 0.1
        mean_df   = float(np.nanmean(diff_frac))
        mean_df_ag = float(np.nanmean(diff_frac[m_ag])) if m_ag.any() else np.nan
        frac_gt02  = float(np.nanmean((diff_frac > 0.2).astype(float)))
        if mean_df < 0.20:    v = "PASS"
        elif mean_df > 0.40:  v = "FAIL"
        else:                  v = "WARN"
        profile = np.nanmean(diff_frac, axis=(1, 2))
        rows: list[tuple[str, str]] = [
            ("Metric",         "|Q_ertel − Q_lin| / |Q_ertel|  [consistent units]"),
            ("Measured",       f"{mean_df:.4f}"),
            ("Acceptable",     "< 0.20"),
            ("Ageost (Ro>0.1)", f"{mean_df_ag:.4f}"),
            ("frac diff>0.2",  f"{frac_gt02:.1%}  (nonlinear effects here)"),
        ]
        note = ("Ertel PV well approximated by QG linearisation; QGPV loss sufficient." if v == "PASS"
                else "Nonlinear Ertel PV differs from QG linearisation; use Ertel loss where Ro>0.1.")
        self._block("EQ-8", "Ertel PV", rows, v, note)
        self._fig_depth("eq8_ertel_pv", dv, profile, 0.20, 0.40, "fraction")
        r = dict(eq="EQ-8", name="Ertel PV",
                 measured=mean_df, acceptable="< 0.20", verdict=v, units="fraction")
        self._results.append(r)
        return r

    # ── EQ-9: salinity conservation ───────────────────────────────────────────

    def _eq9(self) -> dict:
        ds    = self._open_raw()
        t0, t1 = self._30d_window()
        ds30  = ds.isel(time=slice(t0, t1))
        lv    = ds.latitude.values; lov = ds.longitude.values; dv = ds.depth.values
        dt_s  = float((ds.time.values[1] - ds.time.values[0]) / np.timedelta64(1, "s"))
        nlon  = len(lov); nz = len(dv)
        print("  [EQ-9] loading 30d salinity/velocity …", flush=True)
        sp = ds30["so"].compute().values.astype(np.float32)
        pt = ds30["thetao"].compute().values.astype(np.float32)
        uo = ds30["uo"].compute().values.astype(np.float32)
        vo = ds30["vo"].compute().values.astype(np.float32)
        mld = ds30["mlotst"].mean("time").compute().values       # (nlat, nlon)
        p_2d = gsw.p_from_z(-dv[:, None], lv[None, :])          # (nz, nlat)
        p4   = p_2d[None, :, :, None] * np.ones_like(sp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            SA = gsw.SA_from_SP(sp, p4,
                                lov[None, None, None, :] * np.ones_like(sp),
                                lv[None, None, :, None]  * np.ones_like(sp))
            CT = gsw.CT_from_pt(SA, pt)
        wo = -sci_int.cumulative_trapezoid(
            self._dx(uo, lov, lv, 3, 2) + self._dy(vo, lv, 2),
            x=dv, axis=1, initial=0.0)
        KV = 1e-5
        dSA_dt  = np.gradient(SA, dt_s, axis=0)
        dSA_dz  = self._dz(SA, dv, 1)
        residual = (dSA_dt
                    + uo * self._dx(SA, lov, lv, 3, 2)
                    + vo * self._dy(SA, lv, 2)
                    + wo * dSA_dz
                    - KV * np.gradient(dSA_dz, dv, axis=1))
        mld_mean = float(np.nanmean(mld))
        res_bml  = self._zmean(np.abs(residual), dv, mld_mean, dv[-1], zax=1)
        aaiw_res = self._zmean(np.abs(residual), dv, 600, 1200, zax=1)
        # depth at which profile first drops below 1e-7
        profile  = np.nanmean(np.abs(residual), axis=(0, 2, 3))   # (nz,)
        below_th = np.where(profile < 1e-7)[0]
        dep_min  = float(dv[below_th[0]]) if below_th.size else float(dv[-1])
        # term magnitudes
        mag_tend  = float(np.nanmean(np.abs(dSA_dt)))
        mag_adv_h = float(np.nanmean(np.abs(uo * self._dx(SA, lov, lv, 3, 2))
                                    + np.abs(vo * self._dy(SA, lv, 2))))
        mag_adv_z = float(np.nanmean(np.abs(wo * dSA_dz)))
        mag_diff  = float(np.nanmean(np.abs(KV * np.gradient(dSA_dz, dv, axis=1))))
        if res_bml < 1e-7 and aaiw_res < 5e-8:   v = "PASS"
        elif res_bml > 5e-7 or aaiw_res > 2e-7:  v = "FAIL"
        else:                                       v = "WARN"
        rows: list[tuple[str, str]] = [
            ("Metric",       "mean|res| below MLD + AAIW layer"),
            ("Below MLD",    f"{res_bml:.2e} g/kg/s"),
            ("Acceptable",   "below MLD < 1e-7  AND  AAIW < 5e-8 g/kg/s"),
            ("AAIW 600-1200m", f"{aaiw_res:.2e} g/kg/s"),
            ("Reliable below", f"{dep_min:.0f} m  (|res|<1e-7)"),
            ("mag dSA/dt",   f"{mag_tend:.2e}"),
            ("mag horiz adv", f"{mag_adv_h:.2e}"),
            ("mag vert adv", f"{mag_adv_z:.2e}"),
            ("mag diffusion", f"{mag_diff:.2e}"),
        ]
        note = (f"Salinity conserved below {dep_min:.0f}m; set depth_min={dep_min:.0f}m for SA loss."
                if v != "FAIL" else
                "Salinity residual too large; increase depth_min or reduce SA loss weight.")
        self._block("EQ-9", "Salinity conservation", rows, v, note)
        self._fig_depth("eq9_salinity_adv", dv, profile, 1e-7, 5e-7, "g/kg/s")
        r = dict(eq="EQ-9", name="Salinity conservation",
                 measured=res_bml, acceptable="< 1e-7 g/kg/s", verdict=v, units="g/kg/s")
        self._results.append(r)
        return r

    # ── EQ-10: temperature advection-diffusion ────────────────────────────────

    def _eq10(self) -> dict:
        ds    = self._open_raw()
        t0, t1 = self._30d_window()
        ds30  = ds.isel(time=slice(t0, t1))
        lv    = ds.latitude.values; lov = ds.longitude.values; dv = ds.depth.values
        dt_s  = float((ds.time.values[1] - ds.time.values[0]) / np.timedelta64(1, "s"))
        print("  [EQ-10] loading 30d temperature/velocity …", flush=True)
        sp = ds30["so"].compute().values.astype(np.float32)
        pt = ds30["thetao"].compute().values.astype(np.float32)
        uo = ds30["uo"].compute().values.astype(np.float32)
        vo = ds30["vo"].compute().values.astype(np.float32)
        mld = ds30["mlotst"].mean("time").compute().values
        p_2d = gsw.p_from_z(-dv[:, None], lv[None, :])
        p4   = p_2d[None, :, :, None] * np.ones_like(sp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            SA = gsw.SA_from_SP(sp, p4,
                                lov[None, None, None, :] * np.ones_like(sp),
                                lv[None, None, :, None]  * np.ones_like(sp))
            CT = gsw.CT_from_pt(SA, pt)
        wo = -sci_int.cumulative_trapezoid(
            self._dx(uo, lov, lv, 3, 2) + self._dy(vo, lv, 2),
            x=dv, axis=1, initial=0.0)
        KV = 1e-5
        dCT_dt  = np.gradient(CT, dt_s, axis=0)
        dCT_dz  = self._dz(CT, dv, 1)
        residual = (dCT_dt
                    + uo * self._dx(CT, lov, lv, 3, 2)
                    + vo * self._dy(CT, lv, 2)
                    + wo * dCT_dz
                    - KV * np.gradient(dCT_dz, dv, axis=1))
        mld_mean = float(np.nanmean(mld))
        res_bml  = self._zmean(np.abs(residual), dv, mld_mean, dv[-1], zax=1)
        profile  = np.nanmean(np.abs(residual), axis=(0, 2, 3))
        below_th = np.where(profile < 1e-6)[0]
        dep_1e6  = float(dv[below_th[0]]) if below_th.size else float(dv[-1])
        if res_bml < 1e-6:    v = "PASS"
        elif res_bml > 5e-6:  v = "FAIL"
        else:                  v = "WARN"
        rows: list[tuple[str, str]] = [
            ("Metric",        "mean|res| below MLD"),
            ("Below MLD",     f"{res_bml:.2e} °C/s"),
            ("Acceptable",    "< 1e-6 °C/s"),
            ("Depth <1e-6 °C/s", f"{dep_1e6:.0f} m"),
        ]
        note = (f"Temperature conserved below {dep_1e6:.0f}m; set depth_min={dep_1e6:.0f}m."
                if v != "FAIL" else
                "Temperature residual large; surface forcing dominates — apply MLD mask.")
        self._block("EQ-10", "Temperature advection-diffusion", rows, v, note)
        self._fig_depth("eq10_temp_adv", dv, profile, 1e-6, 5e-6, "°C/s")
        r = dict(eq="EQ-10", name="Temperature advection-diffusion",
                 measured=res_bml, acceptable="< 1e-6 °C/s", verdict=v, units="°C/s")
        self._results.append(r)
        return r

    # ── EQ-11: mixed-layer budget ─────────────────────────────────────────────

    def _eq11(self) -> dict:
        ds    = self._open_raw()
        t0, t1 = self._30d_window()
        ds30  = ds.isel(time=slice(t0, t1))
        dt_s  = float((ds.time.values[1] - ds.time.values[0]) / np.timedelta64(1, "s"))
        pt_s  = ds30["thetao"].isel(depth=0).compute().values   # (n30, nlat, nlon)
        mld   = ds30["mlotst"].mean("time").compute().values
        dCT_dt_ml = np.gradient(pt_s, dt_s, axis=0)
        mean_rate = float(np.nanmean(np.abs(dCT_dt_ml)))
        mld_mean  = float(np.nanmean(mld))
        Q_impl    = mean_rate * RHO0 * CP_SEA * mld_mean
        v = "FAIL"
        print(f"\n  {self._SEP}")
        print(f"  EQ-11 | Mixed-layer budget")
        print(f"  {self._SEP}")
        print(f"  {'Verdict':<16}: \033[91mFAIL\033[0m")
        print(f"  EQ-11 requires ERA5 net heat flux (Q_net) to evaluate.")
        print(f"  Without it the residual is entirely the missing forcing term.")
        print(f"  Measured surface dCT/dt = {mean_rate:.2e} °C/s, implying")
        print(f"  Q_net ~ {Q_impl:.1f} W/m² is needed to close the budget.")
        print(f"  Set physics_weight to 0.0 in config until ERA5 is integrated.")
        print(f"  {self._THN}")
        r = dict(eq="EQ-11", name="Mixed-layer budget",
                 measured=mean_rate, acceptable="N/A — needs ERA5", verdict="FAIL",
                 units="°C/s")
        self._results.append(r)
        return r

    # ── runner ────────────────────────────────────────────────────────────────

    def run(self, equations: list[str] | None = None) -> list[dict]:
        """Run all (or selected) equation diagnostics and print the summary."""
        _all: list[tuple[str, object]] = [
            ("1a", self._eq1a), ("1b", self._eq1b), ("2",  self._eq2),
            ("3",  self._eq3),  ("4",  self._eq4),  ("5",  self._eq5),
            ("6",  self._eq6),  ("7",  self._eq7),  ("8",  self._eq8),
            ("9",  self._eq9),  ("10", self._eq10), ("11", self._eq11),
        ]
        sel = dict(_all)
        run_ids = [eid for eid, _ in _all] if equations is None else equations
        print(f"\n{'═'*60}")
        print("  GLORYS12V1 Physics Equation Diagnostics  (PASS / WARN / FAIL)")
        print(f"  Store : {self.store}")
        print(f"{'═'*60}")
        for eid in run_ids:
            if eid in sel:
                try:
                    sel[eid]()
                except Exception as exc:
                    print(f"\n  [EQ-{eid}] ERROR: {exc}")
        self._summary()
        self.save_json()
        return self._results

    # ── summary table ─────────────────────────────────────────────────────────

    def _summary(self) -> None:
        if not self._results:
            return
        H = ("═" * 7, "═" * 30, "═" * 13, "═" * 15, "═" * 8)
        top = "╔" + "╦".join(H) + "╗"
        mid = "╠" + "╬".join(H) + "╣"
        bot = "╚" + "╩".join(H) + "╝"
        hdr = (f"║{'Eq':^7}║{'Name':^30}║{'Measured':^13}"
               f"║{'Acceptable':^15}║{'Verdict':^8}║")
        print(f"\n{top}\n{hdr}\n{mid}")
        vc_m = {"PASS": "\033[92m", "WARN": "\033[93m", "FAIL": "\033[91m"}
        rs   = "\033[0m"
        for r in self._results:
            vc  = vc_m.get(r["verdict"], "")
            eq  = r["eq"][:7].center(7)
            nm  = r["name"][:30].ljust(30)
            ms  = f"{r['measured']:.3e}"[:13].center(13)
            ac  = r["acceptable"][:15].center(15)
            vd  = r["verdict"]
            vds = f"{vc}{vd}{rs}".center(8 + len(vc) + len(rs))
            print(f"║{eq}║{nm}║{ms}║{ac}║{vds}║")
        print(bot)
        # Weight recommendations
        print("\nRecommended loss weights based on verdicts:")
        for r in self._results:
            v = r["verdict"]
            if v == "PASS":   wt = "1.0   — use with recommended weight"
            elif v == "WARN": wt = "0.1   — reduced weight, apply mask"
            else:              wt = "0.0   — do not use (set in config)"
            print(f"  {r['eq']:<6}: {wt}")

    def save_json(self) -> pathlib.Path:
        """Persist results to results/diagnostics/equation_diagnostics.json."""
        import json

        def _ser(v: object) -> object:
            if isinstance(v, (np.floating, np.integer)):
                return float(v)
            return v

        out_dict = {r["eq"]: {k: _ser(vv) for k, vv in r.items()}
                    for r in self._results}
        path = self.out / "equation_diagnostics.json"
        with open(path, "w") as fh:
            json.dump(out_dict, fh, indent=2)
        print(f"\nResults saved → {path}")
        return path


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate physics equations directly in GLORYS12V1.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples\n"
            "  python -m inrpinn.physics.diagnostics --config configs/data_config.yaml\n"
            "  python -m inrpinn.physics.diagnostics --store data/glorys_agulhas.zarr "
            "--equations 1a 1b 5 7\n"
        ),
    )
    p.add_argument("--store", default=None,
                   help="Path to the Zarr store (data/glorys_agulhas.zarr)")
    p.add_argument("--config", default=None,
                   help="Path to data_config.yaml (reads download.output_dir)")
    p.add_argument("--out", default="results/diagnostics",
                   help="Output directory for figures and JSON/CSV")
    p.add_argument("--n-time", type=int, default=None,
                   help="(EquationDiagnostics only) limit time steps")
    p.add_argument(
        "--equations", nargs="+", default=None,
        metavar="EQ",
        help=(
            "Run DiagnosticsRunner (PASS/WARN/FAIL) for selected equations. "
            "Choices: 1a 1b 2 3 4 5 6 7 8 9 10 11. "
            "Omit to run all equations with the runner."
        ),
    )
    p.add_argument(
        "--runner", action="store_true",
        help="Force use of DiagnosticsRunner even when --equations is not given.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    store_path = args.store
    if store_path is None:
        if args.config is None:
            raise SystemExit("Provide --store or --config")
        with open(args.config) as fh:
            cfg = yaml.safe_load(fh)
        store_path = cfg["download"]["output_dir"]

    use_runner = args.runner or (args.equations is not None)

    if use_runner:
        runner = DiagnosticsRunner(store=store_path, out_dir=args.out)
        runner.run(equations=args.equations)
    else:
        diag = EquationDiagnostics(
            store_path=store_path,
            out_dir=args.out,
            n_time=args.n_time,
        )
        diag.run_all()
        diag.save_summary()
