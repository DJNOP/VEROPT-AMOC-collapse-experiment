#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent  # repo root: .../amoc_collapse_exp

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from result_processing import indicators as ind  # your helpers

# Directories with Veros output
CTRL_DIR = BASE_DIR / "no_hosing"
HOSE_DIR = BASE_DIR / "hosing"

# Last output
TIME_INDEX = -1

# AMOC target latitude (for guide line)
AMOC_LAT_TARGET = 26.5  # deg N (RAPID-ish)

# Plot settings
DELTA_SV_CONTOUR_STEP = 1.0    # Sv
DELTA_T_CONTOUR_STEP = 0.1     # degC
DELTA_S_CONTOUR_STEP = 0.05    # psu


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def _symmetric_limits(field: np.ndarray, min_abs: float) -> tuple[float, float]:
    """Return symmetric (vmin, vmax) around zero for a difference field."""
    absmax = np.nanmax(np.abs(field))
    if not np.isfinite(absmax):
        absmax = min_abs
    vmax = max(min_abs, float(np.ceil(absmax)))
    vmin = -vmax
    return vmin, vmax


def _prepare_overturning_section(run_dir: Path):
    """
    Load last-year overturning section (vsf_depth) for a single run.

    Returns
    -------
    lat : 1D np.ndarray
    depth : 1D np.ndarray (positive down)
    field : 2D np.ndarray, shape (depth, lat)
    used_lat : float
        Model latitude nearest to AMOC_LAT_TARGET.
    year_label : str
    """
    nc = ind._find_overturning_nc(run_dir)
    with ind._open_overturning(nc) as ds:
        da = ind._pick_var(ds)  # vsf_depth or equivalent

        tdim = ind._time_dim(da)
        ydim = ind._lat_dim(da)
        zdim = ind._z_dim(da)
        if ydim is None or zdim is None:
            raise ValueError(f"Missing lat/depth dims in {list(da.dims)}")

        # select time index
        if tdim is not None and tdim in da.dims:
            ntime = da.sizes[tdim]
            if TIME_INDEX == -1:
                year_idx = ntime - 1
                year_label = f"year {ntime}"
            else:
                year_idx = TIME_INDEX
                year_label = f"time index {TIME_INDEX}"
            da = da.isel({tdim: year_idx})
        else:
            year_label = "last output"

        # convert to Sv
        fac = ind._units_to_sv(da)
        if fac != 1.0:
            da = da * fac
            da.attrs["units"] = "Sv"

        # coords
        z = np.asarray(da.coords[zdim].values, dtype=float)
        depth = -z if np.nanmedian(z) < 0 else z
        lat = np.asarray(da.coords[ydim].values, dtype=float)

        # shape (depth, lat)
        field = np.asarray(da.transpose(ydim, zdim).values, dtype=float).T

        # nearest latitude to AMOC_LAT_TARGET
        j = int(np.nanargmin(np.abs(lat - AMOC_LAT_TARGET)))
        used_lat = float(lat[j])

    return lat, depth, field, used_lat, year_label


def _find_averages_nc(run_dir: Path) -> Path:
    """Locate the averages file (e.g. global_4deg.averages.nc) in a run dir."""
    explicit = run_dir / "global_4deg.averages.nc"
    if explicit.exists():
        return explicit

    for pat in ("*averages*.nc", "*.averages.nc"):
        hits = sorted(run_dir.glob(pat))
        if hits:
            return hits[0]

    raise FileNotFoundError(f"No averages file found in {run_dir}")


def _prepare_zonal_section(run_dir: Path, varname: str):
    """
    Load last-year zonal-mean section for variable 'varname'
    (e.g. 'temp' or 'salt').

    Returns
    -------
    lat : 1D np.ndarray
    depth : 1D np.ndarray (positive down)
    field : 2D np.ndarray, shape (depth, lat)
    year_label : str
    """
    nc = _find_averages_nc(run_dir)
    with xr.open_dataset(nc, decode_timedelta=True) as ds:
        if varname not in ds:
            raise KeyError(f"'{varname}' not found in {nc.name}")

        da = ds[varname]  # expected dims: Time, zt, yt, xt (order can vary)

        dims = da.dims
        tdim = next((d for d in dims if d.lower().startswith("time")), None)
        zdim = next((d for d in dims if d.lower().startswith("z")), None)
        ydim = next((d for d in dims if d.lower().startswith("y")), None)
        xdim = next((d for d in dims if d.lower().startswith("x")), None)

        if any(d is None for d in (zdim, ydim, xdim)):
            raise ValueError(f"Could not identify (x,y,z) dims from {dims}")

        year_label = None
        if tdim is not None and tdim in dims:
            ntime = da.sizes[tdim]
            if TIME_INDEX == -1:
                year_idx = ntime - 1
                year_label = f"year {ntime}"
            else:
                year_idx = TIME_INDEX
                year_label = f"time index {TIME_INDEX}"
            da = da.isel({tdim: year_idx}).squeeze(drop=True)

        if year_label is None:
            year_label = "last output"

        # zonal mean (skip NaNs over land)
        da_zonal = da.mean(dim=xdim, skipna=True)

        z = np.asarray(da_zonal.coords[zdim].values, dtype=float)
        depth = -z if np.nanmedian(z) < 0 else z
        lat = np.asarray(da_zonal.coords[ydim].values, dtype=float)

        field = np.asarray(da_zonal.transpose(ydim, zdim).values, dtype=float).T

    return lat, depth, field, year_label


# ---------------------------------------------------------------------
# PLOT FUNCTIONS
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
    """Plot vsf_depth difference field (A - B)."""
    fig, ax = plt.subplots(figsize=(13, 5.5))

    vmin, vmax = _symmetric_limits(diff, min_abs=1.0)

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

    # zero contour
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
    ax.set_title(f"Overturning difference ({label_a} – {label_b}) • {year_label}")

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Δ overturning [Sv]")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_zonal_diff(
    lat,
    depth,
    diff,
    used_lat,
    label_a,
    label_b,
    year_label,
    out_path: Path,
    field_label: str,
    min_abs: float,
    contour_step: float,
):
    """Generic zonal-mean difference plot."""
    fig, ax = plt.subplots(figsize=(13, 5.5))

    vmin, vmax = _symmetric_limits(diff, min_abs=min_abs)

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

    cmax = np.ceil(vmax / contour_step) * contour_step
    levels = np.arange(-cmax, cmax + contour_step, contour_step)

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

    # zero contour
    ax.contour(
        lat,
        depth,
        diff,
        levels=[0.0],
        colors="k",
        linewidths=1.2,
    )

    ax.axvline(used_lat, ls="--", c="k", lw=1.1, alpha=0.8)

    ax.invert_yaxis()
    ax.set_xlabel("Latitude (°N)")
    ax.set_ylabel("Depth (m, positive down)")
    ax.set_title(f"Zonal-mean {field_label} difference ({label_a} – {label_b}) • {year_label}")

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(f"Δ {field_label}")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:
    label_ctrl = "no_hosing"
    label_hose = "hosing"

    out_dir = BASE_DIR / "comparison_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Control   dir: {CTRL_DIR}")
    print(f"Hosing    dir: {HOSE_DIR}")
    print(f"Output -> {out_dir}")

    # ----- Overturning -----
    lat_c, depth_c, over_c, used_lat_c, year_label = _prepare_overturning_section(CTRL_DIR)
    lat_h, depth_h, over_h, used_lat_h, _ = _prepare_overturning_section(HOSE_DIR)

    if not (np.allclose(lat_c, lat_h) and np.allclose(depth_c, depth_h)):
        raise ValueError("Latitude/depth grids differ between control and hosing.")

    used_lat = used_lat_c  # they should be nearly identical

    over_diff = over_h - over_c

    # Some AMOC numbers
    max_over_c = float(np.nanmax(over_c))
    max_over_h = float(np.nanmax(over_h))

    j_amoc = int(np.nanargmin(np.abs(lat_c - AMOC_LAT_TARGET)))
    amoc_c = float(np.nanmax(over_c[:, j_amoc]))
    amoc_h = float(np.nanmax(over_h[:, j_amoc]))

    print("\n--- Overturning stats (last year) ---")
    print(f"Max overturning (control)   : {max_over_c:.2f} Sv")
    print(f"Max overturning (hosing)    : {max_over_h:.2f} Sv")
    print(f"AMOC at ~{AMOC_LAT_TARGET:.1f}°N (control): {amoc_c:.2f} Sv")
    print(f"AMOC at ~{AMOC_LAT_TARGET:.1f}°N (hosing) : {amoc_h:.2f} Sv")
    print(f"Δ AMOC (hosing - control)   : {amoc_h - amoc_c:.2f} Sv")

    over_out = out_dir / "overturning_diff_hosing-minus-no_hosing.png"
    _plot_overturning_diff(
        lat_c,
        depth_c,
        over_diff,
        used_lat,
        label_hose,
        label_ctrl,
        year_label,
        over_out,
    )
    print(f"Wrote overturning difference plot: {over_out}")

    # ----- Zonal-mean temperature -----
    lat_tc, depth_tc, temp_c, year_label_t = _prepare_zonal_section(CTRL_DIR, "temp")
    lat_th, depth_th, temp_h, _ = _prepare_zonal_section(HOSE_DIR, "temp")

    if not (np.allclose(lat_tc, lat_th) and np.allclose(depth_tc, depth_th)):
        raise ValueError("Latitude/depth grids differ for temperature.")

    temp_diff = temp_h - temp_c
    temp_out = out_dir / "temp_zonal_diff_hosing-minus-no_hosing.png"
    _plot_zonal_diff(
        lat_tc,
        depth_tc,
        temp_diff,
        used_lat,
        label_hose,
        label_ctrl,
        year_label_t,
        temp_out,
        field_label="temperature [°C]",
        min_abs=0.05,
        contour_step=DELTA_T_CONTOUR_STEP,
    )
    print(f"Wrote temperature difference plot: {temp_out}")

    # ----- Zonal-mean salinity -----
    lat_sc, depth_sc, salt_c, year_label_s = _prepare_zonal_section(CTRL_DIR, "salt")
    lat_sh, depth_sh, salt_h, _ = _prepare_zonal_section(HOSE_DIR, "salt")

    if not (np.allclose(lat_sc, lat_sh) and np.allclose(depth_sc, depth_sh)):
        raise ValueError("Latitude/depth grids differ for salinity.")

    salt_diff = salt_h - salt_c
    salt_out = out_dir / "salt_zonal_diff_hosing-minus-no_hosing.png"
    _plot_zonal_diff(
        lat_sc,
        depth_sc,
        salt_diff,
        used_lat,
        label_hose,
        label_ctrl,
        year_label_s,
        salt_out,
        field_label="salinity [psu]",
        min_abs=0.01,
        contour_step=DELTA_S_CONTOUR_STEP,
    )
    print(f"Wrote salinity difference plot: {salt_out}")

    # Simple surface-mean salinity north of 50N
    nc_ctrl = _find_averages_nc(CTRL_DIR)
    nc_hose = _find_averages_nc(HOSE_DIR)

    with xr.open_dataset(nc_ctrl) as ds_c, xr.open_dataset(nc_hose) as ds_h:
        salt_c_surf = ds_c["salt"].isel(Time=TIME_INDEX, zt=-1)
        salt_h_surf = ds_h["salt"].isel(Time=TIME_INDEX, zt=-1)

        yt = salt_c_surf["yt"]
        mask_n50 = yt >= 50.0

        sc_n50 = salt_c_surf.sel(yt=yt[mask_n50])
        sh_n50 = salt_h_surf.sel(yt=yt[mask_n50])

        mean_c = float(sc_n50.mean().values)
        mean_h = float(sh_n50.mean().values)

        print("\n--- Surface salinity stats north of 50°N (last year) ---")
        print(f"Mean SSS (control) : {mean_c:.3f} psu")
        print(f"Mean SSS (hosing)  : {mean_h:.3f} psu")
        print(f"Δ SSS (hosing - control): {mean_h - mean_c:.3f} psu")


if __name__ == "__main__":
    main()

