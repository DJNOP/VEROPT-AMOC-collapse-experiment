from veropt.interfaces.experiment import ExperimentObjective
from veropt.optimiser.optimiser_saver_loader import load_optimiser_from_state
from veropt.graphical.visualisation import plot_progression_from_optimiser
from veropt.graphical.visualisation import plot_prediction_grid_from_optimiser

optimiser = load_optimiser_from_state(
    file_name="/groups/ocean/nicholas/amoc_collapse_exp/runs/global4deg_amoc-20251130-210803/global4deg_amoc/optimiser_state.json"
)

plot_progression_from_optimiser(
    optimiser)

plot_prediction_grid_from_optimiser(
    optimiser)

