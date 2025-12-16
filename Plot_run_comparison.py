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

# ---------------------------------------------------------------------
# Helpers (moved to Helper_functions.py)
# ---------------------------------------------------------------------

import Helper_functions as hf


def _get_target_lat_from_settings() -> float:
    """
    Pull AMOC target latitude from SETTINGS_PATH.

    This keeps your original behaviour: if the key is missing, we raise.
    """
    settings = hf.read_json(SETTINGS_PATH)
    return float(settings["latlon_boxes"]["AMOC_lat_degN"])


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
    lat_a, depth_a, field_a, used_lat_a, year_label = hf.prepare_overturning_section(
        point_a_dir,
        target_lat=_get_target_lat_from_settings(),
        time_index=TIME_INDEX,
        label_mode="time_index",
    )
    lat_b, depth_b, field_b, used_lat_b, _ = hf.prepare_overturning_section(
        point_b_dir,
        target_lat=_get_target_lat_from_settings(),
        time_index=TIME_INDEX,
        label_mode="time_index",
    )# sanity checks (same grid)
    if not (np.allclose(lat_a, lat_b) and np.allclose(depth_a, depth_b)):
        raise ValueError("Latitude/depth grids differ between the two points.")

    # choose used_lat from A (they should be nearly identical anyway)
    used_lat = used_lat_a

    over_diff = field_a - field_b

    over_out = (
        out_dir / f"overturning_diff_point{POINT_A}-point{POINT_B}.png"
    )
    title = f"vsf_depth difference ({label_a} – {label_b}) • {year_label}"
    hf.plot_overturning_diff(
        lat_a,
        depth_a,
        over_diff,
        used_lat,
        label_a,
        label_b,
        year_label,
        over_out,
        contour_step=DELTA_SV_CONTOUR_STEP,
        title=title,
    )
    print(f"Wrote overturning difference: {over_out}")

    # ----- Zonal-mean temperature sections -----
    lat_t_a, depth_t_a, temp_a, year_label_t = hf.prepare_zonal_section(
        point_a_dir,
        varname="temp",
        time_index=TIME_INDEX,
        label_mode="time_index",
        year_label_hint=year_label,
    )
    lat_t_b, depth_t_b, temp_b, _ = hf.prepare_zonal_section(
        point_b_dir,
        varname="temp",
        time_index=TIME_INDEX,
        label_mode="time_index",
        year_label_hint=year_label,
    )

    # sanity checks
    if not (np.allclose(lat_t_a, lat_t_b) and np.allclose(depth_t_a, depth_t_b)):
        raise ValueError(
            "Latitude/depth grids for temperature differ between the two points."
        )

    temp_diff = temp_a - temp_b

    temp_out = out_dir / f"temp_diff_point{POINT_A}-point{POINT_B}.png"
    title = f"Zonal-mean temperature difference ({label_a} – {label_b}) • {year_label_t}"
    hf.plot_zonal_diff(
        lat_t_a,
        depth_t_a,
        temp_diff,
        used_lat,
        label_a,
        label_b,
        year_label_t,
        temp_out,
        field_label="temperature [°C]",
        cbar_label="Δ temperature [°C]",
        min_abs=0.05,
        contour_step=DELTA_T_CONTOUR_STEP,
        title=title,
        clabel_fmt="%.2f",
    )
    print(f"Wrote temperature difference: {temp_out}")


if __name__ == "__main__":
    main()
