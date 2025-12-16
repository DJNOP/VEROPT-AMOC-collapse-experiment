#!/usr/bin/env python3
"""
Helper_functions.py

Small, reusable utilities shared by the result_processing scripts.

Design goals
- Reduce duplicated boilerplate (file finding, JSON IO, time selection, zonal means, plotting).
- Keep behaviour stable: scripts should produce the same outputs given the same inputs.
- Avoid circular imports: this module must NOT import result_processing.indicators.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Optional, Sequence

import json
import numpy as np
import xarray as xr


# ---------------------------------------------------------------------
# JSON / file I/O
# ---------------------------------------------------------------------

def read_json(path: str | Path) -> Dict[str, Any]:
    """Read a JSON file and return its contents."""
    with open(path, "r") as fh:
        return json.load(fh)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist; return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plots_dir(run_dir: str | Path) -> Path:
    """Return the canonical plots directory under a run directory."""
    return ensure_dir(Path(run_dir) / "plots")


def find_averages_nc(run_dir: Path) -> Path:
    """
    Locate the averages file (e.g. global_4deg.averages.nc) in a directory.

    Search order (matches your scripts):
      1) run_dir/global_4deg.averages.nc
      2) first match in ("*averages*.nc", "*.averages.nc")
    """
    explicit = run_dir / "global_4deg.averages.nc"
    if explicit.exists():
        return explicit

    for pat in ("*averages*.nc", "*.averages.nc"):
        hits = sorted(run_dir.glob(pat))
        if hits:
            return hits[0]

    raise FileNotFoundError(f"No averages file found under {run_dir}")


# ---------------------------------------------------------------------
# Overturning file helpers (kept compatible with indicators.py)
# ---------------------------------------------------------------------

_OVERTURNING_FILE_PATS: Tuple[str, ...] = (
    "overturning*.nc",
    "overturn*.nc",
    "moc*.nc",
    "*MOC*.nc",
    "*overturn*.nc",
)


def find_overturning_nc(run_dir: str | Path) -> Path:
    """Find the first overturning NetCDF file under a run directory."""
    run_dir = Path(run_dir)
    for pat in _OVERTURNING_FILE_PATS:
        hits = sorted(run_dir.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No overturning file found under {run_dir}")


def open_overturning(nc: str | Path) -> xr.Dataset:
    """
    Open an overturning dataset with a robust engine fallback.

    NOTE: returns an xarray Dataset that supports context manager usage:
        with open_overturning(path) as ds: ...
    """
    nc = Path(nc)
    try:
        return xr.open_dataset(nc, engine="h5netcdf", decode_timedelta=True)
    except Exception as e:
        try:
            return xr.open_dataset(nc, decode_timedelta=True)
        except Exception:
            raise ValueError(f"Could not open overturning file: {nc}") from e


def pick_vsf_depth(ds: xr.Dataset) -> xr.DataArray:
    """Pick the overturning variable you use everywhere (vsf_depth)."""
    if "vsf_depth" in ds.data_vars:
        return ds["vsf_depth"]
    raise KeyError("vsf_depth not found in dataset")


def lat_dim_name(_da: xr.DataArray) -> str:
    """Return your canonical latitude dim name for overturning output."""
    return "yu"


def z_dim_name(_da: xr.DataArray) -> str:
    """Return your canonical depth dim name for overturning output."""
    return "zw"


def time_dim_name(_da: xr.DataArray) -> str:
    """Return your canonical time dim name for overturning output."""
    return "Time"


def units_to_sv(da: xr.DataArray) -> float:
    """Convert overturning units to Sv (if possible)."""
    u = (da.attrs.get("units") or "").lower()
    u = u.replace(" ", "").replace("³", "3").replace("−", "-")
    if "sv" in u:
        return 1.0
    if u in {"m3/s", "m^3/s", "m3s-1", "m^3s-1", "m3s^-1", "m^3s^-1"}:
        return 1e-6
    raise ValueError(f"Unrecognized units for overturning: {u}")


def lat_slice_band(da: xr.DataArray, ydim: str, lat_min: float, lat_max: float) -> xr.DataArray:
    """Latitude band selection that respects ascending/descending coordinates."""
    y = np.asarray(da[ydim].values, dtype=float)
    if y.size == 0:
        raise ValueError(f"No coordinates on {ydim}")
    lo, hi = sorted((float(lat_min), float(lat_max)))
    ascending = bool(y[0] <= y[-1])
    return da.sel({ydim: slice(lo, hi) if ascending else slice(hi, lo)})


def lat_select_nearest(da: xr.DataArray, ydim: str, lat_deg: float) -> tuple[xr.DataArray, float]:
    """Select nearest latitude index; return (slice, used_lat)."""
    lat_vals = np.asarray(da[ydim].values, dtype=float)
    if lat_vals.size == 0 or not np.isfinite(lat_vals).any():
        return da, float("nan")
    idx = int(np.nanargmin(np.abs(lat_vals - float(lat_deg))))
    used = float(lat_vals[idx])
    return da.isel({ydim: idx}), used


def safe_depth_window(
    da: xr.DataArray,
    zdim: str,
    z_bounds_m: Optional[Tuple[float, float]],
) -> xr.DataArray:
    """
    Apply a depth window ONLY if z looks like meters (|z| > ~2).
    This matches the defensive behaviour in indicators.py.
    """
    if z_bounds_m is None or zdim not in da.dims:
        return da

    zcoord = da.coords.get(zdim)
    if zcoord is None or zcoord.size == 0:
        return da

    zvals = np.asarray(zcoord.values, dtype=float)
    if not np.isfinite(zvals).any():
        return da

    looks_like_m = bool(np.nanmax(np.abs(zvals)) > 2)
    if not looks_like_m:
        return da

    lo, hi = sorted(map(float, z_bounds_m))
    # Veros often stores depth negative down
    if np.nanmedian(zvals) < 0:
        lo, hi = -hi, -lo

    ascending = bool(zvals[0] <= zvals[-1])
    sl = slice(lo, hi) if ascending else slice(hi, lo)
    sub = da.sel({zdim: sl})
    return da if sub.size == 0 or sub.sizes.get(zdim, 0) == 0 else sub


def select_last_year_annual(da: xr.DataArray, *, tdim: str = "Time") -> tuple[xr.DataArray, int]:
    """Return (last time-slice, used_flag)."""
    if tdim in da.dims and da.sizes.get(tdim, 0) >= 1:
        return da.isel({tdim: -1}), 1
    return da, 0


# ---------------------------------------------------------------------
# Generic xarray dim helpers used by plotting scripts
# ---------------------------------------------------------------------

def _time_dim(da: xr.DataArray) -> Optional[str]:
    """Best-effort find of time-like dimension name."""
    for d in da.dims:
        if d.lower().startswith("time"):
            return d
    return None


def _coord_dim(da: xr.DataArray, prefix: str) -> str:
    """Find the first dim starting with prefix (case-insensitive)."""
    return next(d for d in da.dims if d.lower().startswith(prefix))


def select_time_and_label(
    da: xr.DataArray,
    time_index: int,
    *,
    label_mode: str = "time_index",
) -> Tuple[xr.DataArray, str]:
    """
    Select a time slice and generate the label you used in the plotting scripts.

    label_mode:
      - "time_index":  last -> 'year N', otherwise -> 'time index k'
      - "year_number": last -> 'year N', otherwise -> 'year (k+1)'
    """
    tdim = _time_dim(da)
    if tdim is None or tdim not in da.dims:
        return da, "last output"

    ntime = int(da.sizes[tdim])
    if time_index == -1:
        idx = ntime - 1
        label = f"year {ntime}"
    else:
        idx = int(time_index)
        if label_mode == "year_number":
            label = f"year {idx + 1}"
        else:
            label = f"time index {idx}"

    return da.isel({tdim: idx}), label


def depth_axis_from_z(z: np.ndarray) -> np.ndarray:
    """Convert Veros-style z (often negative down) to positive depth."""
    z = np.asarray(z, dtype=float)
    return (-z) if np.nanmedian(z) < 0 else z


# ---------------------------------------------------------------------
# High-level data extraction (used by multiple scripts)
# ---------------------------------------------------------------------

def prepare_overturning_section(
    run_dir: Path,
    *,
    target_lat: float,
    time_index: int = -1,
    label_mode: str = "time_index",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    """
    Load an overturning section (vsf_depth) for a run/point directory.

    Returns
    -------
    lat : 1D np.ndarray
    depth : 1D np.ndarray (positive down)
    field : 2D np.ndarray with shape (depth, lat)
    used_lat : float
        Model latitude nearest to target_lat.
    label : str
        Year/time label.
    """
    nc = find_overturning_nc(run_dir)
    with open_overturning(nc) as ds:
        da = pick_vsf_depth(ds)

        tdim = _time_dim(da) or time_dim_name(da)
        ydim = _coord_dim(da, "y")  # "yu" typically
        zdim = _coord_dim(da, "z")  # "zw" typically

        if ydim not in da.dims or zdim not in da.dims:
            raise ValueError(f"Missing lat/depth dims in {list(da.dims)}")

        da, label = select_time_and_label(da, time_index, label_mode=label_mode)

        fac = units_to_sv(da)
        if fac != 1.0:
            da = da * fac
            da.attrs["units"] = "Sv"

        z = np.asarray(da.coords[zdim].values, dtype=float)
        depth = depth_axis_from_z(z)

        lat = np.asarray(da.coords[ydim].values, dtype=float)
        field = np.asarray(da.transpose(ydim, zdim).values, dtype=float).T

        # nearest latitude for guide line
        _, used_lat = lat_select_nearest(da, ydim, target_lat)

    return lat, depth, field, used_lat, label


def prepare_zonal_section(
    averages_nc: Path,
    *,
    varname: str,
    time_index: int = -1,
    label_mode: str = "time_index",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Load a zonal-mean section (lat x depth) from an averages file.

    Returns
    -------
    lat : 1D np.ndarray
    depth : 1D np.ndarray (positive down)
    field : 2D np.ndarray, shape (depth, lat)
    label : str
    """
    with xr.open_dataset(averages_nc) as ds:
        if varname not in ds:
            raise KeyError(f"{varname} not in {averages_nc.name}: {list(ds.data_vars)}")
        da = ds[varname]

        # pick dims by prefix
        ydim = _coord_dim(da, "y")
        zdim = _coord_dim(da, "z")

        da, label = select_time_and_label(da, time_index, label_mode=label_mode)

        z = np.asarray(da.coords[zdim].values, dtype=float)
        depth = depth_axis_from_z(z)
        lat = np.asarray(da.coords[ydim].values, dtype=float)

        field = np.asarray(da.transpose(ydim, zdim).values, dtype=float).T

    return lat, depth, field, label


def load_surface_sss(averages_nc: Path, *, time_index: int = -1) -> np.ndarray:
    """Load surface SSS (1D over latitude) from averages file."""
    with xr.open_dataset(averages_nc) as ds:
        if "salt" not in ds:
            raise KeyError(f"'salt' not found in {averages_nc.name}")
        da = ds["salt"]

        # dims: (time, z, y, x) or similar; pick surface (z=0) and zonal mean (x mean)
        if _time_dim(da) in da.dims:
            da = da.isel({_time_dim(da): time_index})
        if any(d.lower().startswith("z") for d in da.dims):
            zdim = _coord_dim(da, "z")
            da = da.isel({zdim: 0})
        if any(d.lower().startswith("x") for d in da.dims):
            xdim = _coord_dim(da, "x")
            da = da.mean(dim=xdim, skipna=True)

        return np.asarray(da.values, dtype=float)


# ---------------------------------------------------------------------
# Plot utilities
# ---------------------------------------------------------------------

def symmetric_limits(field: np.ndarray, min_abs: float) -> tuple[float, float]:
    """Return symmetric (vmin, vmax) around zero for a difference field."""
    absmax = np.nanmax(np.abs(field))
    if not np.isfinite(absmax):
        absmax = min_abs
    vmax = max(min_abs, float(np.ceil(absmax)))
    vmin = -vmax
    return vmin, vmax


def plot_overturning_diff(
    ax,
    lat: np.ndarray,
    depth: np.ndarray,
    diff: np.ndarray,
    *,
    used_lat: float,
    title: str,
    step: float,
    min_abs: float = 1.0,
):
    """Standardised overturning difference plot."""
    vmin, vmax = symmetric_limits(diff, min_abs=min_abs)
    levels = np.arange(vmin, vmax + step, step)

    cs = ax.contourf(lat, depth, diff, levels=levels, extend="both")
    ax.invert_yaxis()
    ax.set_xlabel("Latitude (deg N)")
    ax.set_ylabel("Depth (m)")
    ax.axvline(used_lat, linestyle="--")
    ax.set_title(title)
    return cs


def plot_zonal_diff(
    ax,
    lat: np.ndarray,
    depth: np.ndarray,
    diff: np.ndarray,
    *,
    title: str,
    step: float,
    min_abs: float,
):
    """Standardised zonal difference plot."""
    vmin, vmax = symmetric_limits(diff, min_abs=min_abs)
    levels = np.arange(vmin, vmax + step, step)

    cs = ax.contourf(lat, depth, diff, levels=levels, extend="both")
    ax.invert_yaxis()
    ax.set_xlabel("Latitude (deg N)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(title)
    return cs


# ---------------------------------------------------------------------
# Veropt objective argument parsing (used in indicators.compute_loss_for_run)
# ---------------------------------------------------------------------

def infer_run_dir_and_settings_path(
    args: Sequence[Any],
    kwargs: Dict[str, Any],
    *,
    default_settings_path: str = "configs/amoc_settings.json",
) -> Tuple[str | Path, str | Path]:
    """
    Infer (run_dir, settings_path) from the same calling patterns you used before.

    This is a *behaviour-preserving* extraction of the logic in indicators.compute_loss_for_run.
    """
    settings_path = kwargs.get("settings_path") or default_settings_path
    run_dir = kwargs.get("run_dir") or kwargs.get("path") or kwargs.get("dir")

    if len(args) == 2:
        a, b = args
        if isinstance(a, str) and a.endswith(".json") and Path(a).is_file():
            settings_path, run_dir = a, b
        elif isinstance(b, str) and b.endswith(".json") and Path(b).is_file():
            run_dir, settings_path = a, b
        else:
            run_dir = a if run_dir is None else run_dir
    elif len(args) == 1 and run_dir is None:
        run_dir = args[0]

    if run_dir is None:
        raise ValueError("compute_loss_for_run: could not infer run_dir")

    return run_dir, settings_path
