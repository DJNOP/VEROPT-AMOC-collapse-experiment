from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import xarray as xr

# Import shared helpers
import Helper_functions as hf


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
    zb = settings["latlon_boxes"]["AMOC_z_bounds_m"]
    z_bounds_m = (float(zb[0]), float(zb[1]))
    lat_deg = float(settings["latlon_boxes"]["AMOC_lat_degN"])
    amoc_sign = float(settings.get("amoc_sign", -1.0))  # Veros Atlantic: negative AMOC

    # last annual mean output
    v = da
    used_n_time = 0
    if time_dim in v.dims and v.sizes.get(time_dim, 0) >= 1:
        v, used_n_time = hf.select_last_year_annual(v, tdim=time_dim)

    # depth window
    v = hf.safe_depth_window(v, z_dim, z_bounds_m)
    if v.sizes.get(z_dim, 0) == 0:
        return {
            "amoc_sv": float("nan"),
            "reason": "empty depth window",
            "z_bounds_m": list(z_bounds_m),
        }

    # nearest latitude
    v_lat, used_lat = hf.lat_select_nearest(v, lat_dim, lat_deg)

    # vertical reduction at that latitude
    if amoc_sign < 0:
        val_native = v_lat.min(dim=z_dim, skipna=True)
        depth_reduction = "min"
    else:
        val_native = v_lat.max(dim=z_dim, skipna=True)
        depth_reduction = "max"

    factor = hf.units_to_sv(da)  # units from original array
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
      - (optional) amoc_sign : -1 (Veros Atlantic, default) or +1

    Assumes annual output → uses last sample.
    """
    run_dir = Path(run_dir)
    settings_path = Path(settings_path)

    s = hf.read_json(settings_path)
    target_amoc = float(s["target"]["target_amoc_Sv"])

    nc = hf.find_overturning_nc(run_dir)
    with hf.open_overturning(nc) as ds:
        var = hf.pick_vsf_depth(ds)
        ydim = hf.lat_dim_name(var)   # "yu"
        zdim = hf.z_dim_name(var)     # "zw"
        tdim = hf.time_dim_name(var)  # "Time"

        if ydim is None or zdim is None:
            return {
                "amoc_sv": float("nan"),
                "depth_reduction": "min/max",
                "reason": "missing y or z dimension",
                "settings_path": str(settings_path),
            }

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

    Behaviour: identical input parsing and outputs as your previous version.
    """
    run_dir, settings_path = hf.infer_run_dir_and_settings_path(args, kwargs)

    detail = compute_amoc_detail_from_settings(run_dir, settings_path)
    amoc_sv = float(detail.get("amoc_sv", float("nan")))

    metrics = dict(detail)
    metrics["amoc_sv"] = amoc_sv

    if not np.isfinite(amoc_sv):
        return {
            "loss": float("nan"),      # NaN → OverflowObjFun returns NaN → skip
            "metrics": metrics,
            "note": "AMOC invalid; skip in optimisation",
        }

    return {"loss": float(amoc_sv), "metrics": metrics}
