#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
import Helper_functions as hf

from result_processing import indicators as ind

# ------------------------------------------------------------
RUN_DIR = Path(
    "/groups/ocean/nicholas/amoc_collapse_exp/runs/"
    "global4deg_amoc-20251130-210803"
)
# ------------------------------------------------------------

TARGET_LAT = 26.5
CMAP = "viridis"
LEVELS = 36
TIME_INDEX = -1
SETTINGS_PATH = Path("configs/amoc_settings.json")


def plot_last_year_overturning(point_dir: Path) -> Path:
    lat, depth, field, used_lat, year_label = hf.prepare_overturning_section(
        point_dir,
        target_lat=TARGET_LAT,
        time_index=TIME_INDEX,
        label_mode="year_number",
    )

    j = int(np.nanargmin(np.abs(lat - float(TARGET_LAT))))
    used_lat = float(lat[j])
    k_star = int(np.nanargmax(np.abs(field[:, j])))
    depth_star = float(depth[k_star])

    fig, ax = plt.subplots(figsize=(13, 5.5))
    cf = ax.contourf(lat, depth, field, levels=LEVELS, cmap=CMAP)

    amoc_levels = np.arange(-20.0, -4.0, 2.0)
    cs = ax.contour(lat, depth, field, levels=amoc_levels, colors="k", linewidths=0.7)
    ax.clabel(cs, fmt="%.0f", fontsize=7)

    ax.invert_yaxis()
    ax.axvline(used_lat, ls="--", c="k", lw=1.1, alpha=0.7)
    ax.scatter([used_lat], [depth_star], marker="*", s=120, c="k", zorder=5)

    ax.set_xlabel("Latitude (°N)")
    ax.set_ylabel("Depth (m, positive down)")
    ax.set_title(f"{point_dir.name} • overturning section • {year_label}")

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Overturning [Sv]")

    out = hf.plots_dir(point_dir) / "overturning_last_year.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def make_amoc_timeseries(point_dir: Path) -> tuple[Path, Path]:
    s = hf.read_json(SETTINGS_PATH)

    nc = ind._find_overturning_nc(point_dir)
    ds = ind._open_overturning(nc)
    da = ind._pick_var(ds)

    tdim = ind._time_dim(da)
    ydim = ind._lat_dim(da)
    zdim = ind._z_dim(da)

    ntime = da.sizes.get(tdim, 1) if tdim in da.dims else 1
    vals, used_lat = [], np.nan

    for i in range(ntime):
        v_i = da.isel({tdim: slice(i, i + 1)}) if tdim in da.dims else da
        out = ind._calculate_amoc(v_i, settings=s, lat_dim=ydim, z_dim=zdim, time_dim=tdim)
        vals.append(float(out.get("amoc_sv", np.nan)))
        if np.isnan(used_lat) and "used_lat_deg" in out:
            used_lat = float(out["used_lat_deg"])

    ds.close()

    years = np.arange(1, len(vals) + 1, dtype=int)
    amoc = np.asarray(vals, dtype=float)

    out_csv = point_dir / "amoc_timeseries.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["year", "amoc_Sv"])
        for y, v in zip(years, amoc):
            w.writerow([int(y), float(v)])

    tgt = s.get("target", {}).get("target_amoc_Sv", None)
    sig = s.get("target", {}).get("amoc_sigma_sv", None)

    plt.figure(figsize=(9, 4.2), dpi=150)
    plt.plot(years, amoc, marker="o", linewidth=1.6)
    plt.axhline(0.0, ls="--", lw=1.0, alpha=0.6)

    if tgt is not None and sig is not None:
        try:
            t, sg = float(tgt), float(sig)
            if np.isfinite(t) and np.isfinite(sg) and sg > 0:
                plt.axhspan(t - sg, t + sg, alpha=0.15, label="target ± σ")
                plt.axhline(t, linestyle=":", alpha=0.7)
                plt.legend(loc="best")
        except Exception:
            pass

    lat_lab = f"{used_lat:.1f}°N" if np.isfinite(used_lat) else "target latitude"
    plt.title(f"{point_dir.name} • AMOC @ {lat_lab}")
    plt.xlabel("Model year")
    plt.ylabel("AMOC strength [Sv]")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    out_png = hf.plots_dir(point_dir) / "amoc_timeseries.png"
    plt.savefig(out_png)
    plt.close()
    return out_png, out_csv


def plot_last_year_zonal_temperature(
    point_dir: Path,
    *,
    target_lat: float = TARGET_LAT,
    cmap: str = CMAP,
    levels: int = LEVELS,
    time_index: int = TIME_INDEX,
) -> Path:
    lat, depth, field, year_label = hf.prepare_zonal_section(
        point_dir,
        varname="temp",
        time_index=time_index,
        label_mode="year_number",
    )

    try:
        olat, odepth, ofield, used_lat_star, _ = hf.prepare_overturning_section(
            point_dir,
            target_lat=target_lat,
            time_index=time_index,
            label_mode="year_number",
        )
        j = int(np.nanargmin(np.abs(olat - float(target_lat))))
        used_lat_star = float(olat[j])
        k_star = int(np.nanargmax(np.abs(ofield[:, j])))
        depth_star = float(odepth[k_star])
    except Exception:
        used_lat_star = depth_star = None

    fig, ax = plt.subplots(figsize=(13, 5.5))
    cf = ax.contourf(lat, depth, field, levels=levels, cmap=cmap)
    ax.invert_yaxis()

    if used_lat_star is not None:
        ax.axvline(used_lat_star, ls="--", c="k", lw=1.1, alpha=0.7)
        ax.scatter([used_lat_star], [depth_star], marker="*", s=120, c="k", zorder=5)

    ax.set_xlabel("Latitude (°N)")
    ax.set_ylabel("Depth (m, positive down)")
    ax.set_title(f"{point_dir.name} • zonal-mean temperature • {year_label}")

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Temperature [°C]")

    out = hf.plots_dir(point_dir) / "temp_zonal_last_year.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    base = RUN_DIR / "global4deg_amoc" / "results"
    print(f"Using results directory: {base}")

    for p in sorted(base.glob("point_*")):
        if not p.is_dir():
            continue
        print(f"\n=== {p.name} ===")

        try:
            out = plot_last_year_overturning(p)
            print(f"Wrote overturning: {out}")
        except Exception as e:
            print(f"Skip overturning for {p} → {e}")

        try:
            tout = plot_last_year_zonal_temperature(p)
            print(f"Wrote temperature: {tout}")
        except Exception as e:
            print(f"Skip zonal temperature for {p} → {e}")

        try:
            png, csvp = make_amoc_timeseries(p)
            print(f"Wrote AMOC series: {png} and {csvp}")
        except Exception as e:
            print(f"Skip AMOC series for {p} → {e}")


if __name__ == "__main__":
    main()
