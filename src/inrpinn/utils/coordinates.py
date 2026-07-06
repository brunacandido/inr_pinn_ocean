"""Coordinate normalisation utilities.

All four input dimensions are mapped to [-1, 1] using the domain bounds
stored in the config (normalisation section).  The same scaler is also
persisted to data/processed/scalers.json so inference is reproducible.

  normalise(coords, bounds) -> Tensor   # (lon,lat,depth,time) → [-1,1]
  denormalise(coords, bounds) -> Tensor # inverse transform
  to_rad(lon, lat) -> (lon_rad, lat_rad) # for Coriolis / trig calculations
"""
