#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

BASE = Path("/groups/ocean/nicholas/amoc_collapse_exp/Hosing_test")

CTRL_AVG = BASE / "no_hosing" / "global_4deg.averages.nc"
HOSE_AVG = BASE / "hosing"    / "global_4deg.averages.nc"


import Helper_functions as hf


print("Loading control run SSS ...")
lat_ctrl, sss_ctrl = hf.load_surface_sss(CTRL_AVG)

print("Loading hosing run SSS ...")
lat_hose, sss_hose = hf.load_surface_sss(HOSE_AVG)

if not np.allclose(lat_ctrl, lat_hose):
    raise ValueError("Latitude grids differ between control and hosing runs")

lat = lat_ctrl
dsss = sss_hose - sss_ctrl

print(
    "ΔSSS (hosing - control) last year: "
    "min={:.3f}, max={:.3f}, mean={:.3f} psu".format(
        float(np.nanmin(dsss)), float(np.nanmax(dsss)), float(np.nanmean(dsss))
    )
)

# Separate N/S of 50°N
mask_N = lat >= 50.0
mask_S = lat < 50.0

mean_N = float(np.nanmean(dsss[mask_N]))
mean_S = float(np.nanmean(dsss[mask_S]))

print("Mean ΔSSS north of 50°N : {:.3f} psu".format(mean_N))
print("Mean ΔSSS south of 50°N : {:.3f} psu".format(mean_S))

# Optional: quick plot saved to Hosing_test
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(lat, dsss, "-o", ms=3)
ax.axvline(50.0, ls="--", color="k", label="50°N")
ax.set_xlabel("Latitude (°N)")
ax.set_ylabel("ΔSSS (hosing - control) [psu]")
ax.set_title("Zonal-mean surface ΔSSS (last year)")
ax.legend()
fig.tight_layout()
out = BASE / "comparison_plots" / "surface_dsss_profile.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Wrote profile plot: {out}")
