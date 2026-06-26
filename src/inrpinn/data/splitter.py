"""Train / val / test splitting at the profile level for full-grid GLORYS data.

Concepts
--------
Profile
    All depth observations at a fixed (time, lat, lon) position.
    The selection unit is always the full profile — depth levels are never
    separated across splits.

Two validation modes
--------------------
uniform
    Validation profiles are drawn uniformly at random from the full domain.
contiguous
    Validation comes from ``n_val_squares`` randomly placed spatial boxes.
    All time steps within each box are assigned to validation.
    This tests spatial generalisation (the model never sees those locations).

Surface vs. full-depth
-----------------------
``surface_fraction`` (optional) reserves an additional pool of profiles for
surface-only (depth-index 0) training observations.  These locations are
separate from the full-depth training set and simulate denser surface coverage
(e.g. satellite SST/SSH).  When not set, all training profiles use every depth.

Reproducibility
---------------
The full split is determined by ``seed``.  The same seed on the same dataset
always produces identical train / val / test / surface assignments, regardless
of how many experiments have been run before.

Usage example
-------------
    import xarray as xr
    from inrpinn.data.splitter import GlorysProfileSplitter

    ds = xr.open_zarr("data/glorys_patch.zarr/derived")
    splitter = GlorysProfileSplitter(ds)

    result = splitter.split(
        mode="uniform",
        train_fraction=0.60,
        val_fraction=0.15,
        surface_fraction=0.20,   # optional extra surface-only profiles
        seed=42,
    )
    splitter.print_summary(result)

    # Use the masks to index into your normalised arrays
    train_coords  = coords_np[result.train_mask]
    train_surf    = coords_np[result.train_surf_mask]   # depth=0 only
    val_coords    = coords_np[result.val_mask]
    test_coords   = coords_np[result.test_mask]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import xarray as xr


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class SplitResult:
    """Boolean masks and index arrays over the flat valid-observation array.

    All arrays have the same length N = total non-NaN grid points (after
    applying the ocean / quality mask of the dataset).

    Attributes
    ----------
    train_mask : (N,) bool
        Full-depth training observations (all depth levels for each selected
        profile).
    train_surf_mask : (N,) bool
        Surface-only (depth index 0) training observations drawn from the
        ``surface_fraction`` pool.  All-False when surface_fraction is None.
    val_mask : (N,) bool
        Validation observations (full depth, never split with training).
    test_mask : (N,) bool
        Test observations (full depth).  Profiles not claimed by any other set.
    profile_id : (N,) int64
        Flat profile identifier:
        ``time_idx * n_lat * n_lon + lat_idx * n_lon + lon_idx``.
    i_lon, i_lat, i_dep, i_tim : (N,) int
        Grid index for each valid observation along each dimension.
    lons, lats, depths : (N,) float
        Physical coordinate values (degrees / metres) for each observation.
    info : dict
        Profile counts, fractions, and split metadata for logging/checkpointing.
    """

    train_mask: np.ndarray
    train_surf_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    profile_id: np.ndarray
    i_lon: np.ndarray
    i_lat: np.ndarray
    i_dep: np.ndarray
    i_tim: np.ndarray
    lons: np.ndarray
    lats: np.ndarray
    depths: np.ndarray
    info: dict = field(default_factory=dict)

    # Convenience properties so callers can ask for counts without touching info
    @property
    def n_train_obs(self) -> int:
        return int(self.train_mask.sum())

    @property
    def n_train_surf_obs(self) -> int:
        return int(self.train_surf_mask.sum())

    @property
    def n_val_obs(self) -> int:
        return int(self.val_mask.sum())

    @property
    def n_test_obs(self) -> int:
        return int(self.test_mask.sum())


# ── Splitter class ─────────────────────────────────────────────────────────────

class GlorysProfileSplitter:
    """Split GLORYS full-grid data into train / val / test at the profile level.

    Parameters
    ----------
    ds : xr.Dataset
        GLORYS-derived dataset with CT and SA variables and dimensions
        (time, depth, latitude, longitude).
    variables : tuple[str, ...]
        Variables whose NaN pattern defines the ocean mask.  A grid cell is
        valid if ALL listed variables are non-NaN at that point.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        variables: tuple[str, ...] = ("CT", "SA"),
    ) -> None:
        self._ds        = ds
        self._variables = variables
        self._lons_g    = ds.longitude.values
        self._lats_g    = ds.latitude.values
        self._depths_g  = ds.depth.values
        self._n_lon     = len(self._lons_g)
        self._n_lat     = len(self._lats_g)
        self._n_dep     = len(self._depths_g)
        self._n_tim     = len(ds.time)

        # Build the flat index arrays from the ocean / quality mask
        self._i_lon, self._i_lat, self._i_dep, self._i_tim = self._compute_valid_indices()

        # Profile ID = unique identifier for each (time, lat, lon) column
        self._profile_id = (
            self._i_tim.astype(np.int64) * self._n_lat * self._n_lon
            + self._i_lat.astype(np.int64) * self._n_lon
            + self._i_lon.astype(np.int64)
        )
        self._unique_pids = np.unique(self._profile_id)

    # ── Public API ─────────────────────────────────────────────────────────────

    def split(
        self,
        mode: Literal["uniform", "contiguous"],
        train_fraction: float,
        val_fraction: float,
        surface_fraction: float | None = None,
        seed: int = 42,
        n_val_squares: int = 5,
    ) -> SplitResult:
        """Create a fully reproducible train / val / test / surface split.

        All fractions are relative to the TOTAL number of (time, lat, lon)
        profiles in the dataset.  Profiles not assigned to any set become the
        implicit test set.

        Parameters
        ----------
        mode : "uniform" | "contiguous"
            Validation sampling strategy.
            ``"uniform"``    — draw validation profiles at random from the whole
                               domain.
            ``"contiguous"`` — draw from ``n_val_squares`` randomly placed
                               spatial boxes (all time steps within each box
                               go to validation).
        train_fraction : float
            Fraction of all profiles used as full-depth training.
        val_fraction : float
            Fraction of all profiles used for validation (full depth).
        surface_fraction : float | None
            Optional fraction of profiles reserved for surface-only (depth=0)
            training.  Drawn from the pool not already taken by training or
            validation.  Profiles are never shared with train or val.
            If None, all training observations use the full water column.
        seed : int
            RNG seed.  The same seed + dataset always gives the same split,
            even across different experiments.
        n_val_squares : int
            Number of spatial boxes in ``"contiguous"`` mode (default 5).

        Returns
        -------
        SplitResult
            Boolean masks and index arrays ready to index into flat observation
            arrays (coords_np, targets_np, …).
        """
        self._validate_fractions(train_fraction, val_fraction, surface_fraction)

        rng       = np.random.default_rng(seed)
        n_total   = len(self._unique_pids)

        # ── Step 1: assign validation profiles ────────────────────────────────
        if mode == "uniform":
            val_pids = self._sample_val_uniform(rng, val_fraction)
        else:
            val_pids = self._sample_val_contiguous(rng, val_fraction, n_val_squares)

        # ── Step 2: full-depth training from the non-validation pool ──────────
        non_val = self._unique_pids[~np.isin(self._unique_pids, val_pids)]
        rng.shuffle(non_val)
        n_train    = max(1, round(train_fraction * n_total))
        train_pids = non_val[:n_train]
        leftover   = non_val[n_train:]

        # ── Step 3: surface-only training (optional) ──────────────────────────
        if surface_fraction is not None:
            n_surf    = max(0, round(surface_fraction * n_total))
            surf_pids = leftover[:n_surf]
        else:
            surf_pids = np.array([], dtype=self._unique_pids.dtype)

        # ── Step 4: everything not claimed becomes test ────────────────────────
        claimed   = set(val_pids) | set(train_pids) | set(surf_pids)
        test_pids = self._unique_pids[~np.isin(self._unique_pids, list(claimed))]

        # ── Step 5: build flat boolean masks over valid observations ───────────
        pid = self._profile_id
        train_mask      = np.isin(pid, train_pids)
        val_mask        = np.isin(pid, val_pids)
        test_mask       = np.isin(pid, test_pids)

        # Surface-only mask: profiles from the surface set AND at depth index 0
        is_surface      = self._i_dep == 0
        train_surf_mask = np.isin(pid, surf_pids) & is_surface

        # Sanity: no observation should appear in more than one set
        assert not (train_mask & val_mask).any(),      "train ∩ val is non-empty"
        assert not (train_mask & test_mask).any(),     "train ∩ test is non-empty"
        assert not (val_mask   & test_mask).any(),     "val ∩ test is non-empty"
        assert not (train_surf_mask & train_mask).any(), "surf_train ∩ train is non-empty"
        assert not (train_surf_mask & val_mask).any(),   "surf_train ∩ val is non-empty"

        info = {
            "seed":                   seed,
            "mode":                   mode,
            "train_fraction":         train_fraction,
            "val_fraction":           val_fraction,
            "surface_fraction":       surface_fraction,
            "n_total_profiles":       n_total,
            "n_train_profiles":       len(train_pids),
            "n_train_surf_profiles":  len(surf_pids),
            "n_val_profiles":         len(val_pids),
            "n_test_profiles":        len(test_pids),
            "n_train_obs":            int(train_mask.sum()),
            "n_train_surf_obs":       int(train_surf_mask.sum()),
            "n_val_obs":              int(val_mask.sum()),
            "n_test_obs":             int(test_mask.sum()),
            "actual_train_fraction":  len(train_pids) / n_total,
            "actual_val_fraction":    len(val_pids)   / n_total,
            "actual_surf_fraction":   len(surf_pids)  / n_total,
            "actual_test_fraction":   len(test_pids)  / n_total,
        }

        return SplitResult(
            train_mask      = train_mask,
            train_surf_mask = train_surf_mask,
            val_mask        = val_mask,
            test_mask       = test_mask,
            profile_id      = pid,
            i_lon           = np.asarray(self._i_lon),
            i_lat           = np.asarray(self._i_lat),
            i_dep           = np.asarray(self._i_dep),
            i_tim           = np.asarray(self._i_tim),
            lons            = self._lons_g[self._i_lon],
            lats            = self._lats_g[self._i_lat],
            depths          = self._depths_g[self._i_dep],
            info            = info,
        )

    def print_summary(self, result: SplitResult) -> None:
        """Print a human-readable summary of a SplitResult to stdout."""
        info    = result.info
        n_total = info["n_total_profiles"]
        line    = "─" * 60

        print(line)
        print(f"Profile split  (seed={info['seed']}  mode={info['mode']})")
        print(line)
        print(
            f"  {'Set':<18}  {'Profiles':>10}  {'%':>7}  {'Obs':>12}"
        )
        print(f"  {'-'*18}  {'-'*10}  {'-'*7}  {'-'*12}")

        rows = [
            ("Full-depth train",   info["n_train_profiles"],      result.n_train_obs),
            ("Surface-only train", info["n_train_surf_profiles"],  result.n_train_surf_obs),
            ("Validation",         info["n_val_profiles"],         result.n_val_obs),
            ("Test (holdout)",     info["n_test_profiles"],        result.n_test_obs),
        ]
        for label, n_prof, n_obs in rows:
            pct = 100.0 * n_prof / n_total if n_total > 0 else 0.0
            print(f"  {label:<18}  {n_prof:>10,}  {pct:>6.1f}%  {n_obs:>12,}")

        print(line)
        print(f"  Total profiles:  {n_total:,}")
        print(line)

    # ── Private helpers ────────────────────────────────────────────────────────

    # Computes flat index arrays (i_lon, i_lat, i_dep, i_tim) for all non-NaN
    # ocean grid points by intersecting the NaN masks of all requested variables.
    def _compute_valid_indices(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        arr   = self._ds[self._variables[0]].transpose(
            "longitude", "latitude", "depth", "time"
        ).values
        valid = ~np.isnan(arr)
        for v in self._variables[1:]:
            arr_v = self._ds[v].transpose(
                "longitude", "latitude", "depth", "time"
            ).values
            valid &= ~np.isnan(arr_v)
        return np.where(valid)

    # Decodes profile IDs back to per-profile (lat_idx, lon_idx) arrays so that
    # spatial queries (e.g. box placement) can operate in geographic coordinates.
    def _pid_to_latlon_idx(
        self, pids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        loc   = pids % (self._n_lat * self._n_lon)
        return loc // self._n_lon, loc % self._n_lon  # lat_idx, lon_idx

    # Draws val_fraction of all unique profiles uniformly at random.
    def _sample_val_uniform(
        self, rng: np.random.Generator, val_fraction: float
    ) -> np.ndarray:
        n_val = max(1, round(val_fraction * len(self._unique_pids)))
        pids  = self._unique_pids.copy()
        rng.shuffle(pids)
        return pids[:n_val]

    # Places n_squares spatial boxes so their combined area targets val_fraction
    # of the domain; all profiles (across all times) inside any box go to val.
    def _sample_val_contiguous(
        self,
        rng: np.random.Generator,
        val_fraction: float,
        n_squares: int,
    ) -> np.ndarray:
        lat_min = float(self._lats_g.min())
        lat_max = float(self._lats_g.max())
        lon_min = float(self._lons_g.min())
        lon_max = float(self._lons_g.max())
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min

        # Square half-side (degrees) sized so n_squares boxes cover val_fraction
        # of the domain area on average (assuming uniform profile density).
        target_area = val_fraction / n_squares * lat_range * lon_range
        half_side   = np.sqrt(target_area) / 2.0

        # Clamp so each box always fits inside the domain regardless of size
        half_lat = min(half_side, lat_range / 2.0 * 0.95)
        half_lon = min(half_side, lon_range / 2.0 * 0.95)

        lat_idx, lon_idx = self._pid_to_latlon_idx(self._unique_pids)
        pid_lats = self._lats_g[lat_idx]
        pid_lons = self._lons_g[lon_idx]

        val_pid_set: set[int] = set()
        for _ in range(n_squares):
            c_lat = float(rng.uniform(lat_min + half_lat, lat_max - half_lat))
            c_lon = float(rng.uniform(lon_min + half_lon, lon_max - half_lon))
            in_box = (
                (pid_lats >= c_lat - half_lat) & (pid_lats <= c_lat + half_lat)
                & (pid_lons >= c_lon - half_lon) & (pid_lons <= c_lon + half_lon)
            )
            val_pid_set.update(self._unique_pids[in_box].tolist())

        return np.array(sorted(val_pid_set), dtype=self._unique_pids.dtype)

    # Raises ValueError when the requested fractions are geometrically impossible.
    def _validate_fractions(
        self,
        train_fraction: float,
        val_fraction: float,
        surface_fraction: float | None,
    ) -> None:
        total = train_fraction + val_fraction + (surface_fraction or 0.0)
        if not (0.0 < train_fraction < 1.0):
            raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
        if not (0.0 < val_fraction < 1.0):
            raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
        if surface_fraction is not None and not (0.0 <= surface_fraction < 1.0):
            raise ValueError(f"surface_fraction must be in [0, 1), got {surface_fraction}")
        if total > 1.0:
            raise ValueError(
                f"train_fraction + val_fraction + surface_fraction = {total:.3f} > 1.0 — "
                "no profiles would remain for test."
            )
