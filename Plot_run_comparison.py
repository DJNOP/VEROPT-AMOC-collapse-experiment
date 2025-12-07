#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# ------------------------- CONFIGURATION ------------------------------
# ---------------------------------------------------------------------

# Root directory of the Veropt / Veros run
RUN_DIR = Path(
    "/groups/ocean/nicholas/amoc_collapse_exp/runs/"
    "global4deg_amoc-20251130-210803"
)

# Point numbers to compare (A - B)
POINT_A = 17
POINT_B = 0

# Time index to use: -1 = last output (i.e. last model year)
TIME_INDEX = -1

# Step for contour lines in difference plots
DELTA_SV_CONTOUR_STEP = 1.0       # Sv for overturning difference
DELTA_T_CONTOUR_STEP = 0.05       # degC for temperature difference

# Settings file – used to read AMOC target latitude (AMOC_lat_degN)
SETTINGS_PATH = Path("configs/amoc_settings.json")

# ---------------------------------------------------------------------
# ------------------------- IMPORT HELPERS -----------------------------
# ---------------------------------------------------------------------

from result_processing import indicators as ind


# ---------------------------------------------------------------------
# ----------------------- GENERIC HELPERS ------------------------------
# ---------------------------------------------------------------------

def _load_settings() -> dict:
    """Load JSON settings used elsewhere for AMOC calculation."""
    with open(SETTINGS_PATH, "r") as fh:
        return json.load(fh)


def _get_target_lat_from_settings() -> float:
    """Return AMOC target latitude (deg N) from settings file."""
    s = _load_settings()
    try:
        return float(s["latlon_boxes"]["AMOC_lat_degN"])
    except Exception as e:  # pragma: no cover (defensive)
        raise KeyError(
            "Could not read 'latlon_boxes.AMOC_lat_degN' from "
            f"{SETTINGS_PATH}"
        ) from e


def _prepare_overturning_section(point_dir: Path):
    """
    Load last-year overturning section for a single point.

    Returns
    -------
    lat : 1D np.ndarray
    depth : 1D np.ndarray (positive down)
    field : 2D np.ndarray with shape (depth, lat)
    used_lat : float
        Model latitude nearest to AMOC_lat_degN from settings.
    year_label : str
        Label for the time index (e.g. 'year 50' or 'last output').
    """
    target_lat = _get_target_lat_from_settings()

    nc = ind._find_overturning_nc(point_dir)
    with ind._open_overturning(nc) as ds:
        da = ind._pick_var(ds)  # vsf_depth or equivalent

        tdim = ind._time_dim(da)
        ydim = ind._lat_dim(da)
        zdim = ind._z_dim(da)
        if ydim is None or zdim is None:
            raise ValueError(f"Missing lat/depth dims in {list(da.dims)}")

        # select time
        ntime = da.sizes.get(tdim, 0) if tdim in da.dims else 0
        if tdim in da.dims:
            if TIME_INDEX == -1:
                year_idx = ntime - 1
                year_label = f"year {ntime}"
            else:
                year_idx = TIME_INDEX
                year_label = f"time index {TIME_INDEX}"
            da = da.isel({tdim: year_idx})
        else:
            year_label = "last output"

        # convert to Sv if needed
        fac = ind._units_to_sv(da)
        if fac != 1.0:
            da = da * fac
            da.attrs["units"] = "Sv"

        # coords for plotting
        z = np.asarray(da.coords[zdim].values, dtype=float)
        depth = -z if np.nanmedian(z) < 0 else z
        lat = np.asarray(da.coords[ydim].values, dtype=float)
        field = np.asarray(da.transpose(ydim, zdim).values, dtype=float).T

        # nearest model latitude to target_lat from settings
        j = int(np.nanargmin(np.abs(lat - float(target_lat))))
        used_lat = float(lat[j])

    return lat, depth, field, used_lat, year_label


def _find_averages_nc(point_dir: Path) -> Path:
    """Locate the averages file (e.g. global_4deg.averages.nc) in a point dir."""
    explicit = point_dir / "global_4deg.averages.nc"
    if explicit.exists():
        return explicit

    for pat in ("*averages*.nc", "*.averages.nc"):
        hits = sorted(point_dir.glob(pat))
        if hits:
            return hits[0]

    raise FileNotFoundError(f"No averages file found in {point_dir}")


def _prepare_zonal_temperature(point_dir: Path, year_label_hint: str | None = None):
    """
    Load last-year zonal-mean temperature section for a single point.

    Returns
    -------
    lat : 1D np.ndarray
    depth : 1D np.ndarray (positive down)
    field : 2D np.ndarray with shape (depth, lat)
    year_label : str
        If year_label_hint is given, reuse it; otherwise infer from time dim.
    """
    nc = _find_averages_nc(point_dir)
    with xr.open_dataset(nc, decode_timedelta=True) as ds:
        if "temp" not in ds:
            raise KeyError(f"'temp' not found in {nc.name}")

        temp = ds["temp"]  # expected dims: Time, zt, yt, xt

        dims = temp.dims
        tdim = next((d for d in dims if d.lower().startswith("time")), None)
        zdim = next((d for d in dims if d.lower().startswith("z")), None)
        ydim = next((d for d in dims if d.lower().startswith("y")), None)
        xdim = next((d for d in dims if d.lower().startswith("x")), None)

        if any(d is None for d in (zdim, ydim, xdim)):
            raise ValueError(f"Could not identify (x,y,z) dims from {dims}")

        # select time (last by default)
        year_label = year_label_hint
        if tdim is not None and tdim in dims:
            ntime = temp.sizes[tdim]
            if TIME_INDEX == -1:
                year_idx = ntime - 1
                if year_label is None:
                    year_label = f"year {ntime}"
            else:
                year_idx = TIME_INDEX
                if year_label is None:
                    year_label = f"time index {TIME_INDEX}"
            temp = temp.isel({tdim: year_idx}).squeeze(drop=True)

        if year_label is None:
            year_label = "last output"

        # zonal mean (ocean only: skip NaNs)
        temp_zonal = temp.mean(dim=xdim, skipna=True)

        z = np.asarray(temp_zonal.coords[zdim].values, dtype=float)
        depth = -z if np.nanmedian(z) < 0 else z
        lat = np.asarray(temp_zonal.coords[ydim].values, dtype=float)
        field = np.asarray(temp_zonal.transpose(ydim, zdim).values, dtype=float).T

    return lat, depth, field, year_label


def _symmetric_limits(field: np.ndarray, min_abs: float = 1.0):
    """Return symmetric (vmin, vmax) around zero for a difference field."""
    absmax = np.nanmax(np.abs(field))
    if not np.isfinite(absmax):
        absmax = min_abs
    vmax = max(min_abs, float(np.ceil(absmax)))
    vmin = -vmax
    return vmin, vmax


# ---------------------------------------------------------------------
# -------------------------- PLOT FUNCTIONS ---------------------------
# ---------------------------------------------------------------------

def _plot_overturning_diff(
    lat,
    depth,
    diff,
    used_lat,
    label_a,
    label_b,
    year_label,
    out_path: Path,
):
    """Plot vsf_depth difference field (A - B) with contour lines and zero contour."""
    fig, ax = plt.subplots(figsize=(13, 5.5))

    vmin, vmax = _symmetric_limits(diff, min_abs=1.0)

    # filled contours
    cf = ax.contourf(
        lat,
        depth,
        diff,
        levels=31,
        vmin=vmin,
        vmax=vmax,
        cmap="RdBu_r",
        extend="both",
    )

    # contour lines
    step = DELTA_SV_CONTOUR_STEP
    cmax = np.ceil(vmax / step) * step
    levels = np.arange(-cmax, cmax + step, step)

    cs = ax.contour(
        lat,
        depth,
        diff,
        levels=levels,
        colors="k",
        linewidths=0.6,
        alpha=0.7,
    )
    ax.clabel(cs, fmt="%.1f", fontsize=7)

    # zero contour highlighted
    ax.contour(
        lat,
        depth,
        diff,
        levels=[0.0],
        colors="k",
        linewidths=1.2,
    )

    # AMOC latitude line
    ax.axvline(used_lat, ls="--", c="k", lw=1.1, alpha=0.8)

    ax.invert_yaxis()
    ax.set_xlabel("Latitude (°N)")
    ax.set_ylabel("Depth (m, positive down)")
    ax.set_title(f"vsf_depth difference ({label_a} – {label_b}) • {year_label}")

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Δ overturning [Sv]")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_temperature_diff(
    lat,
    depth,
    diff,
    used_lat,
    label_a,
    label_b,
    year_label,
    out_path: Path,
):
    """Plot zonal-mean temperature difference field (A - B) with contour lines."""
    fig, ax = plt.subplots(figsize=(13, 5.5))

    vmin, vmax = _symmetric_limits(diff, min_abs=0.05)

    # filled contours
    cf = ax.contourf(
        lat,
        depth,
        diff,
        levels=31,
        vmin=vmin,
        vmax=vmax,
        cmap="RdBu_r",
        extend="both",
    )

    # contour lines
    step = DELTA_T_CONTOUR_STEP
    cmax = np.ceil(vmax / step) * step
    levels = np.arange(-cmax, cmax + step, step)

    cs = ax.contour(
        lat,
        depth,
        diff,
        levels=levels,
        colors="k",
        linewidths=0.6,
        alpha=0.7,
    )
    ax.clabel(cs, fmt="%.2f", fontsize=7)

    # zero contour highlighted
    ax.contour(
        lat,
        depth,
        diff,
        levels=[0.0],
        colors="k",
        linewidths=1.2,
    )

    # AMOC latitude line
    ax.axvline(used_lat, ls="--", c="k", lw=1.1, alpha=0.8)

    ax.invert_yaxis()
    ax.set_xlabel("Latitude (°N)")
    ax.set_ylabel("Depth (m, positive down)")
    ax.set_title(
        f"Zonal-mean temperature difference ({label_a} – {label_b}) • {year_label}"
    )

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Δ temperature [°C]")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# ------------------------------- MAIN --------------------------------
# ---------------------------------------------------------------------

def main() -> None:
    base_results = RUN_DIR / "global4deg_amoc" / "results"
    point_a_dir = base_results / f"point_{POINT_A}"
    point_b_dir = base_results / f"point_{POINT_B}"

    label_a = f"point {POINT_A}"
    label_b = f"point {POINT_B}"

    out_dir = RUN_DIR / "comparison_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run dir: {RUN_DIR}")
    print(f"Comparing {label_a} ({point_a_dir})  –  {label_b} ({point_b_dir})")
    print(f"Output -> {out_dir}")

    # ----- Overturning sections -----
    lat_a, depth_a, field_a, used_lat_a, year_label = _prepare_overturning_section(
        point_a_dir
    )
    lat_b, depth_b, field_b, used_lat_b, _ = _prepare_overturning_section(
        point_b_dir
    )

    # sanity checks (same grid)
    if not (np.allclose(lat_a, lat_b) and np.allclose(depth_a, depth_b)):
        raise ValueError("Latitude/depth grids differ between the two points.")

    # choose used_lat from A (they should be nearly identical anyway)
    used_lat = used_lat_a

    over_diff = field_a - field_b

    over_out = (
        out_dir / f"overturning_diff_point{POINT_A}-point{POINT_B}.png"
    )
    _plot_overturning_diff(
        lat_a,
        depth_a,
        over_diff,
        used_lat,
        label_a,
        label_b,
        year_label,
        over_out,
    )
    print(f"Wrote overturning difference: {over_out}")

    # ----- Zonal-mean temperature sections -----
    lat_t_a, depth_t_a, temp_a, year_label_t = _prepare_zonal_temperature(
        point_a_dir, year_label_hint=year_label
    )
    lat_t_b, depth_t_b, temp_b, _ = _prepare_zonal_temperature(
        point_b_dir, year_label_hint=year_label
    )

    # sanity checks
    if not (np.allclose(lat_t_a, lat_t_b) and np.allclose(depth_t_a, depth_t_b)):
        raise ValueError(
            "Latitude/depth grids for temperature differ between the two points."
        )

    temp_diff = temp_a - temp_b

    temp_out = out_dir / f"temp_diff_point{POINT_A}-point{POINT_B}.png"
    _plot_temperature_diff(
        lat_t_a,
        depth_t_a,
        temp_diff,
        used_lat,
        label_a,
        label_b,
        year_label_t,
        temp_out,
    )
    print(f"Wrote temperature difference: {temp_out}")


if __name__ == "__main__":
    main()
