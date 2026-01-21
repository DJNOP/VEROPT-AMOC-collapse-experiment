# build_veros_experiment.py
from __future__ import annotations
"""
Build and run a Veropt + Veros experiment.

- Safe inside/outside SLURM allocations
- Creates a timestamped run directory under ./runs/
- Writes an auto-generated experiment config to ./.build/
- Wires SLURM-backed Veros runner + result processor
- Resumes an existing experiment if possible (per optimiser state)
"""

import os
import sys
import json
import datetime
from pathlib import Path

from veropt.interfaces.experiment import Experiment
from veropt.interfaces.slurm_simulation import SlurmVerosConfig
from custom_slurm_veros_runner import CustomSlurmVerosRunner
from result_processing.veros_result_processor import VerosResultProcessor


# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
ROOT  = Path(__file__).resolve().parent
CONF  = ROOT / "configs"
RUNS  = ROOT / "runs"
BUILD = ROOT / ".build"

RUNS.mkdir(exist_ok=True)
BUILD.mkdir(exist_ok=True)


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def _ensure_file(p: Path, label: str) -> Path:
    if not p.is_file():
        raise FileNotFoundError(f"Missing {label}: {p}")
    return p


def _resolve_settings_path() -> Path:
    """
    Prefer ./configs/amoc_settings.json, fall back to project root.
    """
    primary = CONF / "amoc_settings.json"
    if primary.is_file():
        return primary
    fallback = ROOT / "amoc_settings.json"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError("Could not find amoc_settings.json (tried configs/ and project root)")


def _create_timestamped_run_dir(experiment_name: str) -> Path:
    run_dir = RUNS / f"{experiment_name}-{_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_auto_config(base_cfg_path: Path, run_dir: Path, settings_path: Path) -> Path:
    """
    Load base experiment config, inject run path + provenance, write to ./.build/.
    """
    base = json.loads(base_cfg_path.read_text())
    base["path_to_experiment"] = str(run_dir)
    base["amoc_settings_path"] = str(settings_path)

    out = BUILD / "veros_experiment_config.auto.json"
    out.write_text(json.dumps(base, indent=2))
    return out


def _print_build_summary(*, base: Path, optimiser: Path, slurm: Path, settings: Path,
                         run_dir: Path, auto_cfg: Path) -> None:
    print("[build] Using configs:")
    print(f"  base:         {base}")
    print(f"  optimiser:    {optimiser}")
    print(f"  slurm:        {slurm}")
    print(f"  settings:     {settings}")
    print(f"  run_dir:      {run_dir}")
    print(f"  auto config:  {auto_cfg}")


def _symlink_latest(run_dir: Path, experiment_name: str) -> None:
    """
    Convenience symlink: runs/<experiment>-latest -> current timestamped directory.
    """
    latest = RUNS / f"{experiment_name}-latest"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(run_dir, target_is_directory=True)
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# Build + Run
# --------------------------------------------------------------------------------------
def build_veros_experiment() -> Experiment:
    experiment_name = os.environ.get("EXPERIMENT_NAME", "global4deg_amoc")

    base_cfg  = _ensure_file(CONF / "veros_experiment_config.base.json", "base config")
    opt_cfg   = _ensure_file(CONF / "optimiser_config.json", "optimiser config")
    slurm_cfg = _ensure_file(CONF / "slurm_veros_config.json", "slurm config")
    settings_js = _resolve_settings_path()

    run_dir = _create_timestamped_run_dir(experiment_name)
    auto_cfg = _write_auto_config(base_cfg, run_dir, settings_js)
    output_filename = "veros_output.h5"

    _print_build_summary(base=base_cfg, optimiser=opt_cfg, slurm=slurm_cfg, settings=settings_js,
                         run_dir=run_dir, auto_cfg=auto_cfg)

    _symlink_latest(run_dir, experiment_name)

    runner = CustomSlurmVerosRunner(config=SlurmVerosConfig.load(str(slurm_cfg)))
    processor = VerosResultProcessor(settings_path=str(settings_js), output_filename=output_filename)

    return Experiment.continue_if_possible(
        simulation_runner=runner,
        result_processor=processor,
        experiment_config=str(auto_cfg),
        optimiser_config=str(opt_cfg),
    )


def run_experiment() -> None:
    exp = build_veros_experiment()
    exp.run_experiment()


if __name__ == "__main__":
    try:
        run_experiment()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
