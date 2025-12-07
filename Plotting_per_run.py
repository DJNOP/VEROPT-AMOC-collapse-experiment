#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json, csv

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# >>>>> CHANGE THIS FOR EACH RUN <<<<<
RUN_DIR = Path(
    "/groups/ocean/nicholas/amoc_collapse_exp/runs/"
    "global4deg_amoc-20251130-210803"
)
# ------------------------------------------------------------

# Basic plotting settings
TARGET_LAT = 26.5
CMAP = "viridis"
LEVELS = 36
TIME_INDEX = -1  # -1 = last output

# Path to AMOC settings (used by indicators._calculate_amoc)
SETTINGS_PATH = Path("configs/amoc_settings.json")

# Reuse everything from your indicators module
from result_processing import indicators as ind


def _plots_dir(point_dir: Path) -> Path:
    """Return / create plots/ folder inside this point directory."""
    pdir = point_dir / "plots"
    pdir.mkdir(exist_ok=True)
    return pdir


def _year_label(da, tdim: str | None, time_index: int) -> str:
    """Human-readable year label for titles."""
    if tdim is None or tdim not in da.dims:
        return "last output"
    ntime = da.sizes.get(tdim, 0)
    if not ntime:
        return "last output"
    if time_index == -1 or time_index == ntime - 1:
        return f"year {ntime}"
    else:
        return f"year {time_index + 1}"


# ------------------------------------------------------------
# Overturning section
# ------------------------------------------------------------
def plot_last_year_overturning(point_dir: Path) -> Path:
    """Full overturning section for chosen year, with target-lat line + star."""
    nc = ind._find_overturning_nc(point_dir)
    with ind._open_overturning(nc) as ds:
        da = ind._pick_var(ds)  # vsf_depth

        tdim = ind._time_dim(da)
        ydim = ind._lat_dim(da)
        zdim = ind._z_dim(da)
        if ydim is None or zdim is None:
            raise ValueError(f"Missing lat/depth dims in {list(da.dims)}")

        year_label = _year_label(da, tdim, TIME_INDEX)

        # select time
        if tdim in da.dims:
            da = da.isel({tdim: TIME_INDEX})

        # convert to Sv if needed
        fac = ind._units_to_sv(da)
        if fac != 1.0:
            da = da * fac
            da.attrs["units"] = "Sv"

        # coords for plotting
        z = np.asarray(da.coords[zdim].values, dtype=float)
        depth = -z if np.nanmedian(z) < 0 else z
        lat = np.asarray(da.coords[ydim].values, dtype=float)
        field = np.asarray(
            da.transpose(ydim, zdim).values, dtype=float
        ).T  # (depth, lat)

        # line + star at target latitude
        j = int(np.nanargmin(np.abs(lat - float(TARGET_LAT))))
        used_lat = float(lat[j])
        k_star = int(np.nanargmax(np.abs(field[:, j])))
        depth_star = float(depth[k_star])

    # plot
    fig, ax = plt.subplots(figsize=(13, 5.5))

    cf = ax.contourf(lat, depth, field, levels=LEVELS, cmap=CMAP)

    # extra contour lines for -5 .. -20 Sv at 2 Sv steps
    amoc_levels = np.arange(-20.0, -4.0, 2.0)  # -20, -18, ..., -6
    cs = ax.contour(lat, depth, field, levels=amoc_levels, colors="k", linewidths=0.7)
    ax.clabel(cs, fmt="%.0f", fontsize=7)

    ax.invert_yaxis()

    ax.axvline(used_lat, ls="--", c="k", lw=1.1, alpha=0.7)
    ax.scatter(
        [used_lat],
        [depth_star],
        marker="*",
        s=120,
        edgecolor="k",
        facecolor="gold",
        zorder=5,
    )

    ax.set_title(f"{point_dir.name} • vsf_depth • {year_label}")
    ax.set_xlabel("Latitude (°N)")
    ax.set_ylabel("Depth (m, positive down)")

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Overturning [Sv]")

    out = _plots_dir(point_dir) / "overturning_last_year.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ------------------------------------------------------------
# AMOC time series
# ------------------------------------------------------------
def _load_settings(p: Path) -> dict:
    with open(p, "r") as fh:
        return json.load(fh)


def make_amoc_timeseries(point_dir: Path) -> tuple[Path, Path]:
    """AMOC per model year using EXACT logic from indicators._calculate_amoc."""
    s = _load_settings(SETTINGS_PATH)
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
        out = ind._calculate_amoc(
            v_i, settings=s, lat_dim=ydim, z_dim=zdim, time_dim=tdim
        )
        vals.append(float(out.get("amoc_sv", np.nan)))
        if np.isnan(used_lat) and "used_lat_deg" in out:
            used_lat = float(out["used_lat_deg"])

    ds.close()
    years = np.arange(1, len(vals) + 1, dtype=int)
    amoc = np.asarray(vals, dtype=float)

    # CSV (keep in point root)
    out_csv = point_dir / "amoc_timeseries.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["year", "amoc_Sv"])
        for y, v in zip(years, amoc):
            w.writerow([int(y), float(v)])

    # target band
    tgt = s.get("target", {}).get("target_amoc_Sv", None)
    sig = s.get("target", {}).get("amoc_sigma_sv", None)

    # Plot
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

    out_png = _plots_dir(point_dir) / "amoc_timeseries.png"
    plt.savefig(out_png)
    plt.close()
    return out_png, out_csv


# ------------------------------------------------------------
# Zonal-mean temperature
# ------------------------------------------------------------
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


def plot_last_year_zonal_temperature(
    point_dir: Path,
    *,
    target_lat: float = TARGET_LAT,
    cmap: str = CMAP,
    levels: int = LEVELS,
    time_index: int = TIME_INDEX,
) -> Path:
    """
    Zonal-mean temperature (lat–depth) for chosen year.

    Uses global_4deg.averages.nc and averages temp over longitude.
    Also marks the location of strongest overturning with a star.
    """
    # --- temperature field ---
    nc = _find_averages_nc(point_dir)
    with xr.open_dataset(nc, decode_timedelta=True) as ds:
        if "temp" not in ds:
            raise KeyError(f"'temp' not found in {nc.name}")

        temp = ds["temp"]  # dims: time, z, y, x

        dims = temp.dims
        tdim = next((d for d in dims if d.lower().startswith("time")), None)
        zdim = next((d for d in dims if d.lower().startswith("z")), None)
        ydim = next((d for d in dims if d.lower().startswith("y")), None)
        xdim = next((d for d in dims if d.lower().startswith("x")), None)

        if any(d is None for d in (zdim, ydim, xdim)):
            raise ValueError(f"Could not identify (x,y,z) dims from {dims}")

        year_label = _year_label(temp, tdim, time_index)

        if tdim is not None and tdim in dims:
            temp = temp.isel({tdim: time_index}).squeeze(drop=True)

        temp_zonal = temp.mean(dim=xdim, skipna=True)  # dims: z, y

        z = np.asarray(temp_zonal.coords[zdim].values, dtype=float)
        depth = -z if np.nanmedian(z) < 0 else z
        lat = np.asarray(temp_zonal.coords[ydim].values, dtype=float)
        field = np.asarray(
            temp_zonal.transpose(ydim, zdim).values, dtype=float
        ).T  # (depth, lat)

    # --- strongest overturning location ---
    used_lat_star = depth_star = None
    try:
        oc = ind._find_overturning_nc(point_dir)
        with ind._open_overturning(oc) as ods:
            ova = ind._pick_var(ods)
            tdim_o = ind._time_dim(ova)
            ydim_o = ind._lat_dim(ova)
            zdim_o = ind._z_dim(ova)

            if tdim_o in ova.dims:
                ova = ova.isel({tdim_o: time_index})

            fac = ind._units_to_sv(ova)
            if fac != 1.0:
                ova = ova * fac

            oz = np.asarray(ova.coords[zdim_o].values, dtype=float)
            odepth = -oz if np.nanmedian(oz) < 0 else oz
            olat = np.asarray(ova.coords[ydim_o].values, dtype=float)
            ofield = np.asarray(
                ova.transpose(ydim_o, zdim_o).values, dtype=float
            ).T

            j = int(np.nanargmin(np.abs(olat - float(target_lat))))
            used_lat_star = float(olat[j])
            k_star = int(np.nanargmax(np.abs(ofield[:, j])))
            depth_star = float(odepth[k_star])
    except Exception:
        used_lat_star = depth_star = None

    # --- plot ---
    fig, ax = plt.subplots(figsize=(13, 5.5))
    cf = ax.contourf(lat, depth, field, levels=levels, cmap=cmap)
    ax.invert_yaxis()

    if used_lat_star is not None and depth_star is not None:
        ax.scatter(
            [used_lat_star],
            [depth_star],
            marker="*",
            s=120,
            edgecolor="k",
            facecolor="gold",
            zorder=5,
        )
        ax.axvline(used_lat_star, ls="--", c="k", lw=1.1, alpha=0.7)

    ax.set_title(f"{point_dir.name} • zonal-mean temperature • {year_label}")
    ax.set_xlabel("Latitude (°N)")
    ax.set_ylabel("Depth (m, positive down)")

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Temperature [°C]")

    out = _plots_dir(point_dir) / "temp_zonal_last_year.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ------------------------------------------------------------
# Main driver
# ------------------------------------------------------------
def main() -> None:
    base = RUN_DIR / "global4deg_amoc" / "results"
    print(f"Using results directory: {base}")

    for p in sorted(base.glob("point_*")):
        if not p.is_dir():
            continue
        print(f"\n=== {p.name} ===")

        # Overturning section
        try:
            out = plot_last_year_overturning(p)
            print(f"Wrote overturning: {out}")
        except Exception as e:
            print(f"Skip overturning for {p} → {e}")

        # Zonal-mean temperature
        try:
            tout = plot_last_year_zonal_temperature(p)
            print(f"Wrote temperature: {tout}")
        except Exception as e:
            print(f"Skip zonal temperature for {p} → {e}")

        # AMOC time series
        try:
            png, csvp = make_amoc_timeseries(p)
            print(f"Wrote AMOC series: {png} and {csvp}")
        except Exception as e:
            print(f"Skip AMOC series for {p} → {e}")


if __name__ == "__main__":
    main()
