#!/bin/bash -l
#SBATCH -A ocean
#SBATCH -p aegir
#SBATCH --constraint=v3
#SBATCH --job-name=veropt_veros
#SBATCH --time=23:59:59
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --chdir=.

set -euo pipefail
set -x

echo "===== SLURM JOB STARTED ====="
date
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Work dir: $(pwd)"

# mirror stdout/err to a file too
exec > >(tee -a _bootstrap.log) 2>&1

# ---- Miniconda bootstrap (no modulefile) ----
CROOT="/groups/ocean/nicholas/miniconda3"
export PATH="$CROOT/bin:$PATH"
if [ -f "$CROOT/etc/profile.d/conda.sh" ]; then
  . "$CROOT/etc/profile.d/conda.sh"
else
  echo "FATAL: conda.sh not found at $CROOT/etc/profile.d/conda.sh"
  exit 2
fi

# Unstack any active conda envs cleanly
CS="$(printenv CONDA_SHLVL)"
if [ -n "$CS" ]; then
  while true; do
    CS_NOW="$(printenv CONDA_SHLVL)"
    if [ -z "$CS_NOW" ] || [ "$CS_NOW" -le 0 ]; then
      break
    fi
    conda deactivate || break
  done
fi

# Activate Veros runtime env
conda activate /groups/ocean/nicholas/.conda/envs/veros_nicholas || ( echo "FATAL: cannot activate runtime env"; exit 3 )
hash -r

# Defaults WITHOUT any $(...) expansions (avoid braces)
if [ -z "$(printenv RUN_YEARS)" ]; then RUN_YEARS=1; fi
if [ -z "$(printenv OUT_EVERY_YEARS)" ]; then OUT_EVERY_YEARS=1; fi
if [ -z "$(printenv SLURM_NTASKS)" ]; then SLURM_NTASKS=1; fi
export RUN_YEARS
export OUT_EVERY_YEARS
export SLURM_NTASKS

echo "RUN_YEARS=$RUN_YEARS  OUT_EVERY_YEARS=$OUT_EVERY_YEARS  SLURM_NTASKS=$SLURM_NTASKS"
echo "python: $(command -v python)"
echo "veros : $(command -v veros || echo MISSING)"

VEROS_BIN="/groups/ocean/nicholas/.conda/envs/veros_nicholas/bin/veros"

if [ "$SLURM_NTASKS" -le 1 ]; then
  "$VEROS_BIN" run ./global_4_degree.py --loglevel info
  EC=$?
else
  srun -u -n "$SLURM_NTASKS" "$VEROS_BIN" run ./global_4_degree.py --loglevel info
  EC=$?
fi

echo "Exit code: $EC"
date
echo "===== SLURM JOB FINISHED ====="
exit "$EC"
