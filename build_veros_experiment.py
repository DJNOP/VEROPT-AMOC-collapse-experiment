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
from veropt.interfaces.slurm_simulation import SlurmVerosRunner, SlurmVerosConfig
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
# Environment
# --------------------------------------------------------------------------------------
def _configure_environment() -> None:
    # Reasonable defaults if launched outside SLURM
    os.environ.setdefault("SLURM_JOB_ID", "manual")
    os.environ.setdefault("SLURM_JOB_NAME", "veropt_veros")
    os.environ.setdefault("SLURM_SUBMIT_DIR", str(ROOT))

    # Headless plotting for batch runs
    os.environ.setdefault("MPLBACKEND", "Agg")

    # Avoid leaking user site-packages into the run
    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    # Ensure PYTHONPATH doesn't accidentally shadow local packages
    os.environ.pop("PYTHONPATH", None)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _req_file(p: Path, label: str) -> Path:
    """Ensure a required JSON/config file exists."""
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
    raise FileNotFoundError("Could not find amoc_settings.json in ./configs/ or project root.")


def _timestamped_run_dir(experiment_name: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    rd = RUNS / f"{experiment_name}-{ts}"
    rd.mkdir(parents=True, exist_ok=False) 
    return rd


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


def _print_build_summary(*, base: Path, optimiser: Path, slurm: Path, settings: Path, run_dir: Path, auto_cfg: Path) -> None:
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
        # Best-effort only (filesystems/permissions may vary)
        pass


# --------------------------------------------------------------------------------------
# Build + Run
# --------------------------------------------------------------------------------------
def build_veros_experiment() -> Experiment:
    """
    Assemble and return a ready-to-run Experiment object.

    Steps:
      1) Load base / optimiser / SLURM config files
      2) Resolve amoc_settings.json
      3) Create a fresh timestamped run directory
      4) Write an auto-generated experiment config into ./.build/
      5) Instantiate the SLURM Veros runner and the result processor
      6) Resume experiment if possible, otherwise start a new one
    """
    _configure_environment()

    # Required configuration files
    base_cfg  = _req_file(CONF / "veros_experiment_config.base.json", "base experiment config")
    opt_cfg   = _req_file(CONF / "optimiser_config.json", "optimiser config")
    slurm_cfg = _req_file(CONF / "slurm_veros_config.json", "SLURM Veros config")

    # Settings (for indicators)
    settings_js = _resolve_settings_path()

    # Determine experiment name and indicators filename from base config
    base = json.loads(base_cfg.read_text())
    experiment_name = base.get("experiment_name", "amoc_collapse")
    output_filename = base.get("output_filename", "indicators.json")

    # Create run directory + persist auto config with provenance
    run_dir = _timestamped_run_dir(experiment_name)
    auto_cfg = _write_auto_config(base_cfg, run_dir, settings_js)
    _print_build_summary(base=base_cfg, optimiser=opt_cfg, slurm=slurm_cfg, settings=settings_js,
                         run_dir=run_dir, auto_cfg=auto_cfg)

    # Optional: handy symlink to latest
    _symlink_latest(run_dir, experiment_name)

    # Wire runner + result processor
    runner = SlurmVerosRunner(config=SlurmVerosConfig.load(str(slurm_cfg)))
    processor = VerosResultProcessor(settings_path=str(settings_js), output_filename=output_filename)

    # Resume if possible, otherwise start fresh
    try:
        return Experiment.continue_if_possible(
            simulation_runner=runner,
            result_processor=processor,
            experiment_config=str(auto_cfg),
            optimiser_config=str(opt_cfg),
        )
    except KeyError as e:
        # Some code paths require SLURM_* even in local runs; set and retry once.
        if str(e).strip("'\"") in {"SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_SUBMIT_DIR"}:
            _configure_environment()
            return Experiment.continue_if_possible(
                simulation_runner=runner,
                result_processor=processor,
                experiment_config=str(auto_cfg),
                optimiser_config=str(opt_cfg),
            )
        raise


def run_experiment() -> None:
    exp = build_veros_experiment()
    exp.run_experiment()


# --------------------------------------------------------------------------------------
# Entry
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        run_experiment()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
