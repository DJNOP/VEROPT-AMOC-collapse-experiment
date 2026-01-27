from __future__ import annotations

from pathlib import Path
from typing import Any

import csv
import numpy as np

from veropt.optimiser.optimiser_saver_loader import load_optimiser_from_state


# -----------------------------
# CONFIG
# -----------------------------
RUN_DIR = Path(
    "/groups/ocean/nicholas/amoc_collapse_exp/runs/"
    "global4deg_amoc-20260122-150044/"
    "global4deg_amoc"
)
STATE_FILE = RUN_DIR / "global4deg_amoc_optimiser_state.json"

RUN_TAG = RUN_DIR.parent.name
OUT_DIR = Path("/groups/ocean/nicholas/amoc_collapse_exp/figs") / RUN_TAG / "report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "params_sorted_by_amoc.csv"

NONSENSE_CUTOFF = -50.0  # Sv

DEFAULTS: dict[str, float] = {
    "c_k": 0.1,
    "kappaH_min": 2e-5,
    "eke_c_k": 0.4,
    "eke_c_eps": 0.5,
    "sss_offset": 0.0,
}


def _to_numpy(x: Any) -> np.ndarray:
    """Convert torch tensors / numpy / lists to numpy array."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main() -> None:
    if not STATE_FILE.exists():
        raise FileNotFoundError(f"Could not find optimiser state: {STATE_FILE}")

    optimiser = load_optimiser_from_state(file_name=str(STATE_FILE))

    # Prefer real units for the report table
    X = _to_numpy(getattr(optimiser, "evaluated_variables_real_units", optimiser.evaluated_variable_values.tensor))
    Yfull = _to_numpy(getattr(optimiser, "evaluated_objectives_real_units", optimiser.evaluated_objective_values.tensor))

    if Yfull.ndim == 2:
        y = Yfull[:, 0]
    else:
        y = Yfull.reshape(-1)

    var_names = list(optimiser.objective.variable_names)

    n_points = X.shape[0]
    indices = np.arange(n_points, dtype=int)

    finite = np.isfinite(y)
    sensible = finite & (y >= NONSENSE_CUTOFF)
    nonsense = finite & (y < NONSENSE_CUTOFF)
    nonfinite = ~finite

    sensible_sorted = indices[sensible][np.argsort(y[sensible])[::-1]]
    nonsense_sorted = indices[nonsense][np.argsort(y[nonsense])[::-1]]  
    nonfinite_sorted = indices[nonfinite]  

    sorted_indices = np.concatenate([sensible_sorted, nonsense_sorted, nonfinite_sorted])

    # CSV header
    header = ["row_type", "point_index", "amoc_sv"] + var_names

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)

        # Row 1: defaults
        default_row = ["DEFAULT", "", ""]
        for name in var_names:
            val = DEFAULTS.get(name, "")
            if isinstance(val, float):
                default_row.append(f"{val:.16g}")
            else:
                default_row.append(val)
        w.writerow(header)
        w.writerow(default_row)

        # Data rows
        for i in sorted_indices:
            amoc = y[i]
            if np.isfinite(amoc):
                amoc_s = f"{float(amoc):.16g}"
            else:
                amoc_s = "nan"

            row = ["POINT", int(i), amoc_s]
            for j in range(len(var_names)):
                row.append(f"{float(X[i, j]):.16g}")
            w.writerow(row)

    print("Run tag:", RUN_TAG)
    print("Wrote:", OUT_CSV)
    print(f"Total points: {n_points}")
    print(f"Sensible (>= {NONSENSE_CUTOFF}): {sensible.sum()}")
    print(f"Nonsense (< {NONSENSE_CUTOFF}): {nonsense.sum()}")
    print(f"Non-finite: {nonfinite.sum()}")


if __name__ == "__main__":
    main()
