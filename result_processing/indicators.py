from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import json
import numpy as np
import xarray as xr


# =========================
# JSON / FILE I/O HELPERS
# =========================

def _read_json(p: str | Path) -> Dict[str, Any]:
    with open(p, "r") as fh:
        return json.load(fh)


def _find_overturning_nc(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    _OVERTURNING_FILE = ("overturning*.nc", "overturn*.nc", "moc*.nc", "*MOC*.nc", "*overturn*.nc")
    for pat in _OVERTURNING_FILE:
        hits = sorted(run_dir.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No overturning file found under {run_dir}")


def _open_overturning(nc: Path) -> xr.Dataset:
    try:
        return xr.open_dataset(nc, engine="h5netcdf", decode_timedelta=True)
    except Exception as e:
        try:
            return xr.open_dataset(nc, decode_timedelta=True)
        except Exception:
            raise ValueError(f"Could not open overturning file: {nc}") from e


# =========================
# DATA / DIM HELPERS
# =========================

def _pick_var(ds: xr.Dataset) -> xr.DataArray:
    if "vsf_depth" in ds.data_vars:
        return ds["vsf_depth"]
    raise KeyError("vsf_depth not found in dataset")


def _lat_dim(da): return "yu"
def _z_dim(da): return "zw"
def _time_dim(da): return "Time"


def _units_to_sv(da: xr.DataArray) -> float:
    u = (da.attrs.get("units") or "").lower()
    u = u.replace(" ", "").replace("³", "3").replace("−", "-")
    if "sv" in u:
        return 1.0
    if u in {"m3/s", "m^3/s", "m3s-1", "m^3s-1", "m3s^-1", "m^3s^-1"}:
        return 1e-6
    raise ValueError(f"Unrecognized units for overturning: {u}")


def _lat_slice_band(da: xr.DataArray, ydim: str, lat_min: float, lat_max: float) -> xr.DataArray:
    y = np.asarray(da[ydim].values).astype(float)
    if y.size == 0:
        raise ValueError(f"No coordinates on {ydim}")
    lo, hi = sorted((float(lat_min), float(lat_max)))
    ascending = bool(y[0] <= y[-1])
    return da.sel({ydim: slice(lo, hi) if ascending else slice(hi, lo)})


def _lat_select_nearest(da: xr.DataArray, ydim: str, lat_deg: float) -> tuple[xr.DataArray, float]:
    lat_vals = np.asarray(da[ydim].values).astype(float)
    if lat_vals.size == 0 or not np.isfinite(lat_vals).any():
        return da, float("nan")
    idx = int(np.nanargmin(np.abs(lat_vals - float(lat_deg))))
    used = float(lat_vals[idx])
    return da.isel({ydim: idx}), used


def _safe_depth_window(da: xr.DataArray, zdim: str, z_bounds_m: Optional[Tuple[float, float]]) -> xr.DataArray:
    if z_bounds_m is None or zdim not in da.dims:
        return da
    zcoord = da.coords.get(zdim)
    if zcoord is None or zcoord.size == 0:
        return da
    zvals = np.asarray(zcoord.values).astype(float)
    if not np.isfinite(zvals).any():
        return da
    looks_like_m = bool(np.nanmax(np.abs(zvals)) > 2)
    if not looks_like_m:
        return da
    lo, hi = sorted(map(float, z_bounds_m))
    if np.nanmedian(zvals) < 0:
        lo, hi = -hi, -lo
    ascending = bool(zvals[0] <= zvals[-1])
    sl = slice(lo, hi) if ascending else slice(hi, lo)
    sub = da.sel({zdim: sl})
    return da if sub.size == 0 or sub.sizes.get(zdim, 0) == 0 else sub


def _select_last_year_annual(da: xr.DataArray, tdim: str = "Time") -> tuple[xr.DataArray, int]:
    if tdim in da.dims and da.sizes[tdim] >= 1:
        return da.isel({tdim: -1}), 1
    return da, 0


# =========================
# AMOC CALCULATION
# =========================

def _calculate_amoc(
    da: xr.DataArray,
    settings: Dict[str, Any],
    lat_dim: str = "yu",
    z_dim: str = "zw",
    time_dim: str = "Time",
) -> Dict[str, Any]:
    """
    AMOC (Sv) at a single latitude using parameters from `settings` JSON.

    Uses:
      - settings["latlon_boxes"]["AMOC_z_bounds_m"] = [z_lo, z_hi]
      - settings["latlon_boxes"]["AMOC_lat_degN"]
      - optional settings["amoc_sign"] : -1 for negative AMOC (Veros), +1 if positive

    Steps:
      - take last time index (assumed annual mean),
      - restrict to depth window,
      - for that latitude, take extremum in depth (min if sign<0, max if sign>0),
      - convert to Sv.
    """
    # --- read params from JSON ---
    zb = settings["latlon_boxes"]["AMOC_z_bounds_m"]
    z_bounds_m = (float(zb[0]), float(zb[1]))
    lat_deg = float(settings["latlon_boxes"]["AMOC_lat_degN"])
    amoc_sign = float(settings.get("amoc_sign", -1.0))  # Veros Atlantic: negative AMOC

    # --- time selection: last output (annual mean) ---
    v = da
    used_n_time = 0
    if time_dim in v.dims and v.sizes.get(time_dim, 0) >= 1:
        v, used_n_time = _select_last_year_annual(v, tdim=time_dim)

    # --- depth window ---
    v = _safe_depth_window(v, z_dim, z_bounds_m)
    if v.sizes.get(z_dim, 0) == 0:
        return {
            "amoc_sv": float("nan"),
            "reason": "empty depth window",
            "z_bounds_m": list(z_bounds_m),
        }

    # --- single latitude: nearest to AMOC_lat_degN ---
    v_lat, used_lat = _lat_select_nearest(v, lat_dim, lat_deg)

    # --- vertical reduction at that latitude ---
    if amoc_sign < 0:
        # AMOC expected negative, take most negative (min)
        val_native = v_lat.min(dim=z_dim, skipna=True)
        depth_reduction = "min"
    else:
        # AMOC expected positive, take most positive (max)
        val_native = v_lat.max(dim=z_dim, skipna=True)
        depth_reduction = "max"

    # --- units -> Sv ---
    factor = _units_to_sv(da)  # read units from original array
    amoc_sv = float((val_native * factor).item())

    return {
        "amoc_sv": amoc_sv,
        "used_lat_deg": float(used_lat),
        "depth_reduction": depth_reduction,
        "used_n_time": int(used_n_time),
        "z_bounds_m": [float(z_bounds_m[0]), float(z_bounds_m[1])],
        "lat_mode": "single",
    }


def compute_amoc_detail_from_settings(
    run_dir: str | Path,
    settings_path: str | Path,
) -> Dict[str, Any]:
    """
    Compute RAPID-like AMOC (Sv) using parameters from the JSON file.

    JSON drives:
      - latlon_boxes.AMOC_lat_degN
      - latlon_boxes.AMOC_z_bounds_m
      - (optional) latlon_boxes.AMOC_lat_band_degN
      - (optional) lat_weighting: "cos" (default) or "uniform"

    Depth reduction: extremum_abs over vertical. Assumes annual output → last sample.
    """
    run_dir = Path(run_dir)
    settings_path = Path(settings_path)

    with open(settings_path, "r") as fh:
        s = json.load(fh)

    # targets kept for echo in result
    target_amoc = float(s["target"]["target_amoc_Sv"])

    nc = _find_overturning_nc(run_dir)
    with _open_overturning(nc) as ds:
        var = _pick_var(ds)
        ydim = _lat_dim(var)   # "yu"
        zdim = _z_dim(var)     # "zw"
        tdim = _time_dim(var)  # "Time"

        if ydim is None or zdim is None:
            return {
                "amoc_sv": float("nan"),
                "depth_reduction": "extremum_abs",
                "reason": "missing y or z dimension",
                "settings_path": str(settings_path),
            }

        # Single unified call: JSON controls band vs single-lat, weighting, depths
        out = _calculate_amoc(var, settings=s, lat_dim=ydim, z_dim=zdim, time_dim=tdim)

    out.update({
        "target_amoc_Sv": target_amoc,
        "settings_path": str(settings_path),
    })
    out["source_file"] = str(nc)
    return out


def compute_loss_for_run(*args, **kwargs) -> Dict[str, Any]:
    """
    Objective for Veropt based directly on AMOC strength (Sv).

    - We return the AMOC value in Sv as the optimisation objective.
    - Veropt maximises the objective; since AMOC in Veros is negative,
      maximisation pushes AMOC towards 0 Sv (more collapsed).
    """

    # --- parse args / kwargs (same logic as before) ---
    settings_path = kwargs.get("settings_path") or "configs/amoc_settings.json"
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

    # --- compute AMOC details ---
    detail = compute_amoc_detail_from_settings(run_dir, settings_path)
    amoc_sv = float(detail.get("amoc_sv", float("nan")))

    # --- assemble metrics ---
    metrics = dict(detail)
    metrics["amoc_sv"] = amoc_sv

    if not np.isfinite(amoc_sv):
        return {
            "loss": float("nan"),      # NaN → OverflowObjFun returns NaN → skip
            "metrics": metrics,
            "note": "AMOC invalid; skip in optimisation",
        }

    # Here 'loss' is just the AMOC value in Sv
    return {"loss": float(amoc_sv), "metrics": metrics}


