# result_processing/veros_result_processor.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union
import json
import os

from veropt.interfaces.result_processing import ResultProcessor
from veropt.interfaces.simulation import SimulationResult

from .indicators import compute_loss_for_run

RunLike = Union[SimulationResult, dict, str, Path]


class VerosResultProcessor(ResultProcessor):
    """Compute AMOC for a Veros run, persist metrics, and return 'amoc_sv' as objective."""


    _RUN_DIR_KEYS = (
        "output_directory",
        "result_i_directory",
        "result_directory",
        "run_directory",
        "result_dir",
    )

    # Constructor
    def __init__(self, settings_path: str, output_filename: str = "indicators.json") -> None:
        super().__init__(objective_names=["amoc_sv"])
        self.settings_path = settings_path
        self.output_filename = output_filename

    # Method to get output file path
    def open_output_file(self, result: RunLike) -> str:
        """Return the path where this processor will write its JSON metrics."""
        return str(Path(self._run_dir_from_result(result)) / self.output_filename)

    # Helper to extract run directory
    @classmethod
    def _run_dir_from_result(cls, result: RunLike) -> str:
        """Best-effort extraction of the run directory from multiple result shapes."""
        if isinstance(result, (str, Path)):
            return str(result)

        if isinstance(result, dict):
            for k in cls._RUN_DIR_KEYS:
                v = result.get(k)
                if v:
                    return str(v)
        else:
            for k in cls._RUN_DIR_KEYS:
                if hasattr(result, k):
                    v = getattr(result, k)
                    if v:
                        return str(v)

        return os.getcwd()

    # Main method to calculate objectives
    def calculate_objectives(self, result: RunLike) -> Dict[str, float]:
        """
        Return {'amoc_sv': amoc_value}.

        Veropt will maximise 'amoc_sv'. Since AMOC in Veros is negative,
        maximisation pushes the AMOC upwards towards 0 Sv (collapse).
        On failure returns NaN to signal a skip.
        """
        run_dir = self._run_dir_from_result(result)

        try:
            # Compute AMOC + metrics from the JSON-driven pipeline
            out: Dict[str, Any] = compute_loss_for_run(
                run_dir=run_dir,
                settings_path=self.settings_path,
            )

            # Persist the full payload (including metrics)
            self._persist_json(Path(run_dir) / self.output_filename, out)

            # By construction, out["loss"] is now the AMOC in Sv
            amoc_sv = float(out.get("loss", float("nan")))
        except Exception:
            amoc_sv = float("nan")

        # The key name here is the objective name Veropt will show in plots
        return {"amoc_sv": amoc_sv}


    # Helper to persist JSON output
    @staticmethod
    def _persist_json(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
