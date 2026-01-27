from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from veropt.optimiser.optimiser_saver_loader import load_optimiser_from_state
from veropt.graphical.visualisation import (
    plot_progression,
    plot_prediction_grid,
    plot_prediction_surface_grid,
)

# Use the lower-level overview plotter (avoids the "points='best'" assert issue)
from veropt.graphical._overview import plot_point_overview_separate_subplots


# -----------------------------
# Paths
# -----------------------------
RUN_DIR = Path(
    "/groups/ocean/nicholas/amoc_collapse_exp/runs/"
    "global4deg_amoc-20260122-150044/global4deg_amoc"
)
RUN_TAG = RUN_DIR.parent.name  # global4deg_amoc-20260122-150044

OUT_DIR = Path("/groups/ocean/nicholas/amoc_collapse_exp/figs") / RUN_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = RUN_DIR / "global4deg_amoc_optimiser_state.json"


# -----------------------------
# Load optimiser
# -----------------------------
if not STATE_FILE.exists():
    raise FileNotFoundError(f"Could not find optimiser state file: {STATE_FILE}")

optimiser = load_optimiser_from_state(file_name=str(STATE_FILE))


# -----------------------------
# Standard VerOpt plots
# -----------------------------
progression_fig = plot_progression(optimiser)
prediction_grid_fig = plot_prediction_grid(optimiser)
prediction_surface_grid_fig = plot_prediction_surface_grid(
    optimiser,
    objective=optimiser.objective.objective_names[0],
)


# -----------------------------
# Point overview plots (filtered + best10)
# IMPORTANT: keep these as torch tensors (veropt expects .detach())
# -----------------------------
excluded_points = {74, 132, 133, 341}

X = optimiser.evaluated_variables_real_units          # torch.Tensor
Y = optimiser.evaluated_objectives_real_units         # torch.Tensor

# single objective -> use first column
if Y.ndim == 2:
    y_t = Y[:, 0]
else:
    y_t = Y

y = y_t.detach().cpu().numpy()  # for filtering/sorting only

n_points = y.shape[0]
all_indices = np.arange(n_points)

# Keep: finite objective + not excluded
keep_mask = np.isfinite(y)
for p in excluded_points:
    if 0 <= p < n_points:
        keep_mask[p] = False

keep_indices = all_indices[keep_mask].tolist()

# Best10 among kept points:
# With your objective setup, "best" means highest value (closer to 0 / less negative)
sorted_keep = sorted(keep_indices, key=lambda i: float(y[i]), reverse=True)
best10_indices = sorted_keep[:10]

point_overview_filtered_fig = plot_point_overview_separate_subplots(
    variable_values=X,
    objective_values=Y,
    objective_names=optimiser.objective.objective_names,
    variable_names=optimiser.objective.variable_names,
    shown_indices=[int(i) for i in keep_indices],
)

point_overview_best10_fig = plot_point_overview_separate_subplots(
    variable_values=X,
    objective_values=Y,
    objective_names=optimiser.objective.objective_names,
    variable_names=optimiser.objective.variable_names,
    shown_indices=[int(i) for i in best10_indices],
)


# -----------------------------
# Save
# -----------------------------
progression_fig.write_html(str(OUT_DIR / "progression.html"))
prediction_grid_fig.write_html(str(OUT_DIR / "prediction_grid.html"))
prediction_surface_grid_fig.write_html(str(OUT_DIR / "prediction_surface_grid.html"))

point_overview_filtered_fig.write_html(str(OUT_DIR / "point_overview_filtered.html"))
point_overview_best10_fig.write_html(str(OUT_DIR / "point_overview_best10.html"))

print("Run tag:", RUN_TAG)
print("Wrote:", OUT_DIR)
print("Excluded points:", sorted(excluded_points))
print("Best10 indices:", best10_indices)
print("Best10 objective values:", [float(y[i]) for i in best10_indices])
