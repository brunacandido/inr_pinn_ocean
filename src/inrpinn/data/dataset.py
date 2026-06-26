"""Dataset treatment: GLORYS12V1 → sparse Argo-like profiles → train / val / test.

Data flow
---------
1. Load GLORYS zarr (CT, SA, coordinates)
2. Simulate Argo float trajectories; sample GLORYS CT/SA at those positions
3. Normalise coordinates to [-1, 1] and targets by mean/std from the full field
4. Split at the *profile* level: all depth obs from one float at one time step
   land in the same split — prevents depth-level leakage within a water column
5. Return ArgoProfileDataset objects for train, val, and test

Profile-based split
-------------------
A profile = all observations collected by one virtual float at one time step.
Depth levels within a profile are tightly correlated (same water column), so a
point-level random split would leak the vertical structure from train into test.
Here, whole profiles are assigned to one split.  The assignment is random
(seeded, not spatial or temporal) so profiles from anywhere in the domain and
any part of the time series can appear in any split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset


# ── Normalisation helpers ────────────────────────────────────────────────────────

def norm_minmax(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linear map [lo, hi] → [-1, 1]."""
    return 2.0 * (v - lo) / (hi - lo) - 1.0


def denorm_minmax(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (v + 1.0) * (hi - lo) / 2.0 + lo


def norm_zscore(v: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (v - mean) / std


def denorm_zscore(v: np.ndarray, mean: float, std: float) -> np.ndarray:
    return v * std + mean


# ── Dataset class ────────────────────────────────────────────────────────────────

class ArgoProfileDataset(Dataset):
    """Sparse Argo-like ocean profile observations as a PyTorch Dataset.

    Each sample is a pair:
        coords  — (lon, lat, depth, time) normalised to [-1, 1],  shape (4,)
        targets — (CT, SA) z-score normalised,                    shape (2,)
    """

    def __init__(self, coords: np.ndarray, targets: np.ndarray) -> None:
        self.coords  = torch.from_numpy(coords.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.coords[idx], self.targets[idx]


# ── Argo-like sparse sampling ─────────────────────────────────────────────────────

def sample_argo_profiles(
    ds: xr.Dataset,
    reg: dict,
    n_floats: int = 4,
    cycle_days: int = 10,
    depth_coverage: float = 0.85,
    missing_frac: float = 0.10,
    drift_std_deg: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate Argo float trajectories and sample GLORYS CT / SA.

    Parameters
    ----------
    ds : xarray.Dataset
        GLORYS dataset with CT, SA and dims (time, depth, latitude, longitude).
    reg : dict
        Region bounds: lon_min, lon_max, lat_min, lat_max, depth_max.
    n_floats : int
        Number of virtual Argo floats placed in the box.
    cycle_days : int
        Days between profiles per float (Argo standard is 10 days).
    depth_coverage : float
        Fraction of depth levels sampled per profile (partial profile, like real Argo).
    missing_frac : float
        Fraction of sampled levels set to NaN (simulated QC gaps).
    drift_std_deg : float
        Standard deviation of float position drift per cycle [degrees].
    seed : int
        RNG seed; controls initial positions, drift, and depth-level selection.

    Returns
    -------
    pd.DataFrame with columns:
        float_id, lon, lat, depth, time_idx, time_frac, CT, SA

        float_id + time_idx together identify a unique profile (one water column).
    """
    rng = np.random.default_rng(seed)

    lons_g   = ds.longitude.values
    lats_g   = ds.latitude.values
    depths_g = ds.depth.values
    n_times  = len(ds.time)
    n_depths = len(depths_g)

    records: list[dict] = []

    for float_id in range(n_floats):
        lon_f = float(rng.uniform(reg["lon_min"] + 0.3, reg["lon_max"] - 0.3))
        lat_f = float(rng.uniform(reg["lat_min"] + 0.3, reg["lat_max"] - 0.3))

        t_offset  = int(rng.integers(0, cycle_days))
        t_indices = np.arange(t_offset, n_times, cycle_days)

        for t_idx in t_indices:
            lon_f += float(rng.normal(0, drift_std_deg))
            lat_f += float(rng.normal(0, drift_std_deg))
            lon_f  = float(np.clip(lon_f, reg["lon_min"] + 0.1, reg["lon_max"] - 0.1))
            lat_f  = float(np.clip(lat_f, reg["lat_min"] + 0.1, reg["lat_max"] - 0.1))

            i_lon = int(np.argmin(np.abs(lons_g - lon_f)))
            i_lat = int(np.argmin(np.abs(lats_g - lat_f)))

            n_sample  = max(1, int(n_depths * depth_coverage))
            depth_idx = np.sort(rng.choice(n_depths, n_sample, replace=False))

            CT_prof = ds["CT"].values[t_idx, depth_idx, i_lat, i_lon].copy()
            SA_prof = ds["SA"].values[t_idx, depth_idx, i_lat, i_lon].copy()

            gap_mask           = rng.random(n_sample) < missing_frac
            CT_prof[gap_mask]  = np.nan
            SA_prof[gap_mask]  = np.nan

            for k, d_idx in enumerate(depth_idx):
                if np.isnan(CT_prof[k]) or np.isnan(SA_prof[k]):
                    continue
                records.append(
                    {
                        "float_id" : float_id,
                        "lon"      : float(lons_g[i_lon]),
                        "lat"      : float(lats_g[i_lat]),
                        "depth"    : float(depths_g[d_idx]),
                        "time_idx" : int(t_idx),
                        "time_frac": float(t_idx) / max(n_times - 1, 1),
                        "CT"       : float(CT_prof[k]),
                        "SA"       : float(SA_prof[k]),
                    }
                )

    return pd.DataFrame(records)


# ── Profile-level split ───────────────────────────────────────────────────────────

def profile_split(
    df: pd.DataFrame,
    coords: np.ndarray,
    targets: np.ndarray,
    test_frac: float = 0.20,
    val_frac: float  = 0.10,
    seed: int = 42,
) -> tuple[ArgoProfileDataset, ArgoProfileDataset, ArgoProfileDataset, dict]:
    """Split train / val / test at the profile level.

    Whole profiles (all depth obs from one float at one time) are assigned to
    exactly one split.  Selection is random (seeded).

    Parameters
    ----------
    df : pd.DataFrame
        Raw observations; must have float_id and time_idx columns.
    coords : np.ndarray, shape (N, 4)
        Normalised coordinates aligned row-wise with df.
    targets : np.ndarray, shape (N, 2)
        Normalised (CT, SA) targets aligned row-wise with df.
    test_frac : float
        Fraction of profiles reserved for test.
    val_frac : float
        Fraction of profiles reserved for validation.
    seed : int
        Controls which profiles land in which split.

    Returns
    -------
    train_ds, val_ds, test_ds : ArgoProfileDataset
    split_info : dict
        Profile assignments and counts (train / val / test) for logging.
    """
    rng = np.random.default_rng(seed)

    # Map each (float_id, time_idx) → row indices in df
    grouped = df.groupby(["float_id", "time_idx"], sort=False)
    profile_to_idx: dict[tuple, list[int]] = {
        key: list(grp.index) for key, grp in grouped
    }
    profile_keys = list(profile_to_idx.keys())
    rng.shuffle(profile_keys)

    n        = len(profile_keys)
    n_test   = max(1, round(test_frac * n))
    n_val    = max(1, round(val_frac  * n))
    n_train  = n - n_test - n_val

    if n_train < 1:
        raise ValueError(
            f"test_frac={test_frac} + val_frac={val_frac} leaves no training profiles "
            f"(total profiles={n})."
        )

    test_keys  = profile_keys[:n_test]
    val_keys   = profile_keys[n_test : n_test + n_val]
    train_keys = profile_keys[n_test + n_val :]

    def _gather(keys: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
        idx = np.concatenate([profile_to_idx[k] for k in keys]).astype(int)
        return coords[idx], targets[idx]

    train_ds = ArgoProfileDataset(*_gather(train_keys))
    val_ds   = ArgoProfileDataset(*_gather(val_keys))
    test_ds  = ArgoProfileDataset(*_gather(test_keys))

    split_info = {
        "seed"              : seed,
        "n_profiles"        : n,
        "n_train_profiles"  : len(train_keys),
        "n_val_profiles"    : len(val_keys),
        "n_test_profiles"   : len(test_keys),
        "n_train_obs"       : len(train_ds),
        "n_val_obs"         : len(val_ds),
        "n_test_obs"        : len(test_ds),
        "train_profiles"    : [(int(k[0]), int(k[1])) for k in train_keys],
        "val_profiles"      : [(int(k[0]), int(k[1])) for k in val_keys],
        "test_profiles"     : [(int(k[0]), int(k[1])) for k in test_keys],
    }

    return train_ds, val_ds, test_ds, split_info


# ── High-level factory ─────────────────────────────────────────────────────────────

def build_datasets(
    zarr_path: str | Path,
    reg: dict,
    *,
    n_floats: int       = 4,
    cycle_days: int     = 10,
    depth_coverage: float = 0.85,
    missing_frac: float = 0.10,
    drift_std_deg: float = 0.15,
    test_frac: float    = 0.20,
    val_frac: float     = 0.10,
    seed: int           = 42,
) -> tuple[ArgoProfileDataset, ArgoProfileDataset, ArgoProfileDataset, dict]:
    """Full pipeline: GLORYS zarr → normalised, split PyTorch datasets.

    Parameters
    ----------
    zarr_path : path-like
        Root of the GLORYS zarr store; the function opens the ``derived`` group.
    reg : dict
        Region definition: lon_min, lon_max, lat_min, lat_max, depth_max.
    seed : int
        Single seed that controls both the Argo simulation and the profile split,
        guaranteeing full experiment reproducibility.

    Returns
    -------
    train_ds, val_ds, test_ds : ArgoProfileDataset
    meta : dict
        Normalisation parameters (coord_bounds, target_stats), split_info,
        and sampling configuration — save this alongside the model checkpoint
        so inference can invert the normalisation.

    Example
    -------
    >>> from inrpinn.data.dataset import build_datasets
    >>> train, val, test, meta = build_datasets(
    ...     "data/glorys_agulhas.zarr",
    ...     reg={"lon_min": 15, "lon_max": 20, "lat_min": -42,
    ...          "lat_max": -37, "depth_max": 1000},
    ...     seed=42,
    ... )
    """
    zarr_path = Path(zarr_path)
    ds = xr.open_zarr(zarr_path / "derived")

    # Spatial / depth subset
    ds = ds.sel(
        longitude=ds.longitude[
            (ds.longitude >= reg["lon_min"]) & (ds.longitude <= reg["lon_max"])
        ],
        latitude=ds.latitude[
            (ds.latitude >= reg["lat_min"]) & (ds.latitude <= reg["lat_max"])
        ],
        depth=ds.depth[ds.depth <= reg["depth_max"]],
    )

    n_times = len(ds.time)
    coord_bounds: dict[str, tuple[float, float]] = {
        "lon"  : (float(reg["lon_min"]),   float(reg["lon_max"])),
        "lat"  : (float(reg["lat_min"]),   float(reg["lat_max"])),
        "depth": (0.0,                     float(reg["depth_max"])),
        "time" : (0.0,                     float(n_times - 1)),
    }

    # Target statistics from the *full* GLORYS field for stable normalisation
    CT_all = ds["CT"].values.ravel()
    SA_all = ds["SA"].values.ravel()
    CT_all = CT_all[~np.isnan(CT_all)]
    SA_all = SA_all[~np.isnan(SA_all)]
    target_stats: dict[str, float] = {
        "CT_mean": float(CT_all.mean()), "CT_std": float(CT_all.std()),
        "SA_mean": float(SA_all.mean()), "SA_std": float(SA_all.std()),
    }

    # Simulate Argo-like sparse profiles
    df = sample_argo_profiles(
        ds, reg,
        n_floats=n_floats,
        cycle_days=cycle_days,
        depth_coverage=depth_coverage,
        missing_frac=missing_frac,
        drift_std_deg=drift_std_deg,
        seed=seed,
    )

    if df.empty:
        raise RuntimeError(
            "No valid Argo-like observations could be sampled.  "
            "Check that the zarr store and region are correct."
        )

    # Build normalised arrays
    coords_np = np.stack(
        [
            norm_minmax(df.lon.values,       *coord_bounds["lon"]),
            norm_minmax(df.lat.values,       *coord_bounds["lat"]),
            norm_minmax(df.depth.values,     *coord_bounds["depth"]),
            norm_minmax(df.time_idx.values,  *coord_bounds["time"]),
        ],
        axis=1,
    ).astype(np.float32)

    targets_np = np.stack(
        [
            norm_zscore(df.CT.values, target_stats["CT_mean"], target_stats["CT_std"]),
            norm_zscore(df.SA.values, target_stats["SA_mean"], target_stats["SA_std"]),
        ],
        axis=1,
    ).astype(np.float32)

    # Profile-level random split
    train_ds, val_ds, test_ds, split_info = profile_split(
        df, coords_np, targets_np,
        test_frac=test_frac,
        val_frac=val_frac,
        seed=seed,
    )

    meta = {
        "coord_bounds"  : coord_bounds,
        "target_stats"  : target_stats,
        "split_info"    : split_info,
        "n_obs_total"   : len(df),
        "n_profiles"    : df.groupby(["float_id", "time_idx"]).ngroups,
        "sampling"      : {
            "n_floats"      : n_floats,
            "cycle_days"    : cycle_days,
            "depth_coverage": depth_coverage,
            "missing_frac"  : missing_frac,
            "drift_std_deg" : drift_std_deg,
            "seed"          : seed,
        },
    }

    ds.close()
    return train_ds, val_ds, test_ds, meta
