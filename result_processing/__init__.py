"""
result_processing package

Contains indicator computations and a result processor used by Veropt/Veros runs.
"""
from .indicators import compute_loss_for_run, compute_amoc_detail_from_settings  # noqa: F401
from .veros_result_processor import VerosResultProcessor  # noqa: F401
