from pathlib import Path

from veropt.optimiser.optimiser_saver_loader import load_optimiser_from_state
from veropt.graphical.visualisation import (
    plot_progression,
    plot_prediction_grid,
    plot_prediction_surface_grid,
)

RUN_DIR = Path("/groups/ocean/nicholas/amoc_collapse_exp/runs/global4deg_amoc-20260104-162248/global4deg_amoc")

RUN_TAG = RUN_DIR.parent.name

OUT_DIR = Path("figs") / RUN_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

optimiser = load_optimiser_from_state(
    file_name=str(RUN_DIR / "optimiser_state.json")
)

progression_fig = plot_progression(optimiser)
prediction_grid_fig = plot_prediction_grid(optimiser)
prediction_surface_grid_fig = plot_prediction_surface_grid(
    optimiser,
    objective=optimiser.objective.objective_names[0],
)

progression_fig.write_html(str(OUT_DIR / "progression.html"))
prediction_grid_fig.write_html(str(OUT_DIR / "prediction_grid.html"))
prediction_surface_grid_fig.write_html(str(OUT_DIR / "prediction_surface_grid.html"))

print("Run:", RUN_TAG)
print("Wrote:", OUT_DIR)
