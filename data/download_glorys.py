#!/usr/bin/env python3
"""Generalised GLORYS12V1 downloader via Copernicus Marine to Zarr.

Data is streamed in monthly chunks via copernicusmarine.open_dataset() and
appended directly to a single Zarr store — no intermediate NetCDF files.

Usage
-----
    uv run data/download_glorys.py --config configs/data_config.yaml

    # Supply credentials from a file (overrides env vars for that run):
    uv run data/download_glorys.py --config configs/data_config.yaml \\
        --credentials configs/credentials.env

    # Override the time window without editing the config:
    uv run data/download_glorys.py --config configs/data_config.yaml \\
        --start 2020-01-01 --end 2020-12-31

Credentials (evaluated in order — first match wins)
----------------------------------------------------
    1. --credentials <file>  KEY=VALUE file (see configs/credentials.env.example)
    2. COPERNICUSMARINE_SERVICE_USERNAME / _PASSWORD environment variables
    3. Cached interactive login (run once: copernicusmarine login)

Store layout
------------
    {output_dir}/
        raw/       GLORYS12V1 variables (thetao, so, uo, vo, …)
        derived/   Placeholder — populated by scripts/preprocess.py
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from datetime import date
from typing import Any

import pandas as pd
import xarray as xr
import yaml
from zarr.codecs import ZstdCodec


# ── Credentials ────────────────────────────────────────────────────────────────

# Reads a KEY=VALUE credentials file and injects the values into the environment
# so copernicusmarine can pick them up without hardcoding them in the script.
def load_credentials(path: str | pathlib.Path) -> None:
    """Read KEY=VALUE lines from *path* into os.environ.

    Existing environment variables take precedence (os.environ.setdefault).
    Lines starting with # and blank lines are ignored.
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Credentials file not found: {p}")
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


# ── Chunk generation ───────────────────────────────────────────────────────────

# Splits a date range into consecutive monthly windows so the download loop
# can request one manageable chunk at a time instead of the full period at once.
def generate_month_chunks(
    start: date, end: date, chunk_months: int
) -> list[tuple[date, date]]:
    """Split [start, end] into consecutive closed intervals of chunk_months months."""
    chunks: list[tuple[date, date]] = []
    cs = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while cs <= end_ts:
        ce = min(
            cs + pd.DateOffset(months=chunk_months) - pd.DateOffset(days=1),
            end_ts,
        )
        chunks.append((cs.date(), ce.date()))
        cs = cs + pd.DateOffset(months=chunk_months)
    return chunks


# ── Zarr encoding ──────────────────────────────────────────────────────────────

# Builds the Zarr encoding dict for each variable in the dataset, mapping
# dimension names to the configured chunk sizes and attaching the compressor.
def build_encoding(
    ds: xr.Dataset,
    chunk_cfg: dict[str, int],
    compressor: Any,
) -> dict[str, dict]:
    """Build per-variable Zarr encoding from config chunk sizes."""
    chunk_map: dict[str, int] = {
        "time":      chunk_cfg["time"],
        "depth":     chunk_cfg["depth"],
        "latitude":  chunk_cfg["latitude"],
        "longitude": chunk_cfg["longitude"],
    }
    encoding: dict[str, dict] = {}
    for var in ds.data_vars:
        chunks = tuple(chunk_map.get(dim, -1) for dim in ds[var].dims)
        encoding[var] = {"chunks": chunks, "compressors": compressor}
    return encoding


# ── Utilities ──────────────────────────────────────────────────────────────────

# Recursively sums the size of every file under path, used to report how much
# disk space the downloaded Zarr store occupies.
def disk_size_bytes(path: pathlib.Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total


# Converts a byte count to a readable string (MB or GB) for log output.
def human_size(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f} GB"
    return f"{n / 1e6:.1f} MB"


# Returns a horizontal rule string for separating sections in log output.
def _hr() -> str:
    return "─" * 60


# ── Downloader class ───────────────────────────────────────────────────────────

class GlorysDownloader:
    """Download GLORYS data from Copernicus Marine Service to a Zarr store.

    Designed to be reusable for any geographic domain and time period.

    Parameters
    ----------
    dataset_id:
        Copernicus Marine dataset identifier.
    variables:
        List of variable names to request.
    domain:
        Dict with keys lon_min, lon_max, lat_min, lat_max, depth_min, depth_max.
    zarr_chunks:
        Dict with keys time, depth, latitude, longitude (chunk sizes).
    chunk_months:
        Number of months per streaming request. Larger = fewer requests but
        more memory per request.
    clevel:
        Zstandard compression level (1–22).
    name:
        Human-readable label for this region, used in log output.
    """

    # Stores all download parameters and initialises the Zstandard compressor.
    def __init__(
        self,
        dataset_id: str,
        variables: list[str],
        domain: dict[str, float],
        zarr_chunks: dict[str, int],
        chunk_months: int = 1,
        clevel: int = 5,
        name: str = "",
    ) -> None:
        self.dataset_id = dataset_id
        self.variables = variables
        self.domain = domain
        self.zarr_chunks = zarr_chunks
        self.chunk_months = chunk_months
        self.compressor = ZstdCodec(level=clevel)
        self.name = name

    # Alternative constructor: parses a YAML config file and forwards the
    # relevant fields to __init__, so callers don't need to unpack the dict manually.
    @classmethod
    def from_config(cls, config_path: str | pathlib.Path) -> "GlorysDownloader":
        """Instantiate from a YAML config file."""
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)
        dl = cfg.get("download", {})
        return cls(
            dataset_id=cfg["dataset_id"],
            variables=cfg["variables"],
            domain=cfg["domain"],
            zarr_chunks=dl["zarr_chunks"],
            chunk_months=int(dl.get("chunk_months", 1)),
            clevel=int(dl.get("clevel", 5)),
            name=cfg.get("name", ""),
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    # Main entry point: splits the requested period into monthly chunks, fetches
    # each one from Copernicus Marine, and appends it to the Zarr store on disk.
    def download(
        self,
        start: date,
        end: date,
        output_dir: str | pathlib.Path,
    ) -> list[tuple[date, date, str]]:
        """Stream data to *output_dir* and return a list of failed chunks.

        The Zarr store is written to ``{output_dir}/raw`` and a placeholder
        ``{output_dir}/derived`` group is created for downstream processing.

        Parameters
        ----------
        start, end:
            Inclusive date range to download.
        output_dir:
            Root path of the output Zarr store.

        Returns
        -------
        list of (chunk_start, chunk_end, error_message) for any failed chunks.
        """
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        chunks = generate_month_chunks(start, end, self.chunk_months)

        self._print_header(start, end, chunks, output_dir)

        # Deferred import keeps --help fast even if copernicusmarine is slow.
        import copernicusmarine  # noqa: PLC0415

        failed: list[tuple[date, date, str]] = []
        store_initialized = False
        t0 = time.time()

        for i, (cs, ce) in enumerate(chunks):
            n_days = (ce - cs).days + 1
            print(
                f"[{i + 1}/{len(chunks)}]  {cs} → {ce}"
                f"  ({n_days} day{'s' if n_days != 1 else ''})",
                flush=True,
            )
            try:
                ds = copernicusmarine.open_dataset(
                    dataset_id=self.dataset_id,
                    variables=self.variables,
                    minimum_longitude=self.domain["lon_min"],
                    maximum_longitude=self.domain["lon_max"],
                    minimum_latitude=self.domain["lat_min"],
                    maximum_latitude=self.domain["lat_max"],
                    minimum_depth=self.domain["depth_min"],
                    maximum_depth=self.domain["depth_max"],
                    start_datetime=str(cs),
                    end_datetime=str(ce),
                )
                ds = ds.chunk(self.zarr_chunks)
                ds = ds.drop_encoding()

                if not store_initialized:
                    encoding = build_encoding(ds, self.zarr_chunks, self.compressor)
                    ds.to_zarr(
                        output_dir,
                        group="raw",
                        mode="w",
                        encoding=encoding,
                        consolidated=False,
                        align_chunks=True,
                    )
                    store_initialized = True
                else:
                    ds.to_zarr(
                        output_dir,
                        group="raw",
                        append_dim="time",
                        consolidated=False,
                        align_chunks=True,
                    )

                print("        ✓ written")

            except Exception as exc:  # noqa: BLE001
                print(f"        ✗ FAILED: {exc}")
                failed.append((cs, ce, str(exc)))

        self._finalize(output_dir, store_initialized, chunks, failed, t0)
        return failed

    # ── Private helpers ────────────────────────────────────────────────────────

    # Prints a formatted summary of the download job (dataset, domain, period,
    # variable list, output path) before the first chunk is fetched.
    def _print_header(
        self,
        start: date,
        end: date,
        chunks: list[tuple[date, date]],
        output_dir: pathlib.Path,
    ) -> None:
        d = self.domain
        label = self.name or self.dataset_id
        print(_hr())
        print(f"GLORYS download — {label}")
        print(_hr())
        print(f"  Dataset   : {self.dataset_id}")
        print(
            f"  Domain    : lon [{d['lon_min']}, {d['lon_max']}]"
            f"  lat [{d['lat_min']}, {d['lat_max']}]"
            f"  depth [{d['depth_min']}–{d['depth_max']} m]"
        )
        print(f"  Period    : {start} → {end}")
        print(f"  Chunks    : {len(chunks)} × {self.chunk_months} month(s)")
        print(f"  Variables : {self.variables}")
        print(f"  Output    : {output_dir}/raw")
        print(_hr())
        print()

    # Consolidates Zarr metadata, verifies the written store, prints a completion
    # summary with elapsed time and disk size, and lists any chunks that failed.
    def _finalize(
        self,
        output_dir: pathlib.Path,
        store_initialized: bool,
        chunks: list[tuple[date, date]],
        failed: list[tuple[date, date, str]],
        t0: float,
    ) -> None:
        print()
        elapsed = time.time() - t0

        if store_initialized:
            print("Finalising Zarr store …", flush=True)
            try:
                import zarr  # noqa: PLC0415

                root = zarr.open_group(str(output_dir), mode="a")
                root.require_group("derived")
                zarr.consolidate_metadata(str(output_dir))
                print("  Metadata consolidated  ✓")
            except Exception as exc:  # noqa: BLE001
                print(f"  Note: metadata consolidation failed ({exc}) — store is still valid")

            try:
                ds_check = xr.open_zarr(output_dir, group="raw", consolidated=False)
                t_first = pd.Timestamp(ds_check.time.values[0]).strftime("%Y-%m-%d")
                t_last  = pd.Timestamp(ds_check.time.values[-1]).strftime("%Y-%m-%d")
                n_times = int(ds_check.sizes["time"])
                vars_in = sorted(ds_check.data_vars)
                ds_check.close()
            except Exception as exc:  # noqa: BLE001
                t_first = t_last = "unknown"
                n_times = -1
                vars_in = []
                print(f"  Note: could not verify store ({exc})")

            size_str = human_size(disk_size_bytes(output_dir))

            print()
            print(_hr())
            print("Download complete")
            print(_hr())
            print(f"  Elapsed    : {elapsed / 60:.1f} min")
            print(f"  Time range : {t_first} → {t_last}  ({n_times} time steps)")
            print(f"  Variables  : {vars_in}")
            print(f"  Disk size  : {size_str}")
            print(f"  Store      : {output_dir}")
            print(f"    ├── raw/")
            print(f"    └── derived/   (empty — populated by scripts/preprocess.py)")
        else:
            print(_hr())
            print("No chunks were downloaded successfully — store was not created.")

        if failed:
            print()
            print(f"Failed chunks: {len(failed)}/{len(chunks)}")
            for cs, ce, err in failed:
                msg = err[:120] + ("…" if len(err) > 120 else "")
                print(f"  {cs} → {ce}: {msg}")
            print()
            print(
                "To retry the missing period, re-run with --start/--end set to the\n"
                "failed range.  Note: the store will be overwritten from scratch.\n"
                "If most chunks succeeded, pass only the failed date range."
            )
        else:
            n = len(chunks)
            print(f"\nAll {n} chunk{'s' if n != 1 else ''} downloaded successfully.")


# ── CLI ────────────────────────────────────────────────────────────────────────

# Defines and parses the command-line flags: config path, optional credentials
# file, and optional date overrides for start/end.
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download GLORYS data to Zarr via Copernicus Marine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The --start/--end flags override config time.start/time.end without\n"
            "modifying the file.  --credentials loads KEY=VALUE pairs into the\n"
            "environment before the download begins."
        ),
    )
    p.add_argument("--config", required=True, metavar="PATH", help="Path to data config YAML")
    p.add_argument("--credentials", default=None, metavar="PATH",
                   help="Path to credentials env file (default: configs/credentials.env)")
    p.add_argument("--start", default=None, metavar="YYYY-MM-DD",
                   help="Override time.start from config")
    p.add_argument("--end", default=None, metavar="YYYY-MM-DD",
                   help="Override time.end from config")
    return p.parse_args()


# CLI entry point: loads credentials, reads the config, resolves the date range,
# builds a GlorysDownloader, and kicks off the download.
def main() -> None:
    args = parse_args()

    # Load credentials before anything else touches copernicusmarine.
    cred_path = args.credentials or pathlib.Path("configs/credentials.env")
    if pathlib.Path(cred_path).exists():
        load_credentials(cred_path)

    cfg_path = pathlib.Path(args.config)
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    try:
        start = date.fromisoformat(args.start or cfg["time"]["start"])
        end   = date.fromisoformat(args.end   or cfg["time"]["end"])
    except ValueError as exc:
        sys.exit(f"ERROR: invalid date format — {exc}")

    if end < start:
        sys.exit(f"ERROR: --end ({end}) is before --start ({start})")

    downloader = GlorysDownloader.from_config(cfg_path)
    output_dir = pathlib.Path(cfg["download"]["output_dir"])

    downloader.download(start, end, output_dir)


if __name__ == "__main__":
    main()
