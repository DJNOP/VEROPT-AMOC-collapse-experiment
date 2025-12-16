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
# ---------------------------------------------------------------------
# HELPERS (moved to Helper_functions.py)
# ---------------------------------------------------------------------

import Helper_functions as hf


def main() -> None:
    label_ctrl = "no_hosing"
    label_hose = "hosing"

    out_dir = BASE_DIR / "comparison_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Control   dir: {CTRL_DIR}")
    print(f"Hosing    dir: {HOSE_DIR}")
    print(f"Output -> {out_dir}")

    # ----- Overturning -----
    lat_c, depth_c, over_c, used_lat_c, year_label = hf.prepare_overturning_section(
        CTRL_DIR,
        target_lat=AMOC_LAT_TARGET,
        time_index=TIME_INDEX,
        label_mode="time_index",
    )
    lat_h, depth_h, over_h, used_lat_h, _ = hf.prepare_overturning_section(
        HOSE_DIR,
        target_lat=AMOC_LAT_TARGET,
        time_index=TIME_INDEX,
        label_mode="time_index",
    )

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
    title = f"Overturning difference ({label_hose} – {label_ctrl}) • {year_label}"
    hf.plot_overturning_diff(
        lat_c,
        depth_c,
        over_diff,
        used_lat,
        label_hose,
        label_ctrl,
        year_label,
        over_out,
        contour_step=DELTA_SV_CONTOUR_STEP,
        title=title,
    )
    print(f"Wrote overturning difference plot: {over_out}")

    # ----- Zonal-mean temperature -----
    lat_tc, depth_tc, temp_c, year_label_t = hf.prepare_zonal_section(
        CTRL_DIR,
        varname="temp",
        time_index=TIME_INDEX,
        label_mode="time_index",
    )
    lat_th, depth_th, temp_h, _ = hf.prepare_zonal_section(
        HOSE_DIR,
        varname="temp",
        time_index=TIME_INDEX,
        label_mode="time_index",
    )

    if not (np.allclose(lat_tc, lat_th) and np.allclose(depth_tc, depth_th)):
        raise ValueError("Latitude/depth grids differ for temperature.")

    temp_diff = temp_h - temp_c
    temp_out = out_dir / "temp_zonal_diff_hosing-minus-no_hosing.png"
    title = f"Zonal-mean temperature [°C] difference ({label_hose} – {label_ctrl}) • {year_label_t}"
    hf.plot_zonal_diff(
        lat_tc,
        depth_tc,
        temp_diff,
        used_lat,
        label_hose,
        label_ctrl,
        year_label_t,
        temp_out,
        field_label="temperature [°C]",
        cbar_label="Δ temperature [°C]",
        min_abs=0.05,
        contour_step=DELTA_T_CONTOUR_STEP,
        title=title,
        clabel_fmt="%.2f",
    )
    print(f"Wrote temperature difference plot: {temp_out}")

    # ----- Zonal-mean salinity -----
    lat_sc, depth_sc, salt_c, year_label_s = hf.prepare_zonal_section(
        CTRL_DIR,
        varname="salt",
        time_index=TIME_INDEX,
        label_mode="time_index",
    )
    lat_sh, depth_sh, salt_h, _ = hf.prepare_zonal_section(
        HOSE_DIR,
        varname="salt",
        time_index=TIME_INDEX,
        label_mode="time_index",
    )

    if not (np.allclose(lat_sc, lat_sh) and np.allclose(depth_sc, depth_sh)):
        raise ValueError("Latitude/depth grids differ for salinity.")

    salt_diff = salt_h - salt_c
    salt_out = out_dir / "salt_zonal_diff_hosing-minus-no_hosing.png"
    title = f"Zonal-mean salinity [psu] difference ({label_hose} – {label_ctrl}) • {year_label_s}"
    hf.plot_zonal_diff(
        lat_sc,
        depth_sc,
        salt_diff,
        used_lat,
        label_hose,
        label_ctrl,
        year_label_s,
        salt_out,
        field_label="salinity [psu]",
        cbar_label="Δ salinity [psu]",
        min_abs=0.01,
        contour_step=DELTA_S_CONTOUR_STEP,
        title=title,
        clabel_fmt="%.2f",
    )
    print(f"Wrote salinity difference plot: {salt_out}")

    # Simple surface-mean salinity north of 50N
    nc_ctrl = hf.find_averages_nc(CTRL_DIR)
    nc_hose = hf.find_averages_nc(HOSE_DIR)

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
