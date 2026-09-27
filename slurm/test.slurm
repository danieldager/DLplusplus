#!/bin/bash
#SBATCH --job-name=vtc_tests
#SBATCH --output=logs/tests/pytest_%j.out
#SBATCH --error=logs/tests/pytest_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
# Set your partition here, or export SBATCH_PARTITION (it overrides this line).
#SBATCH --partition=<partition>

echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""

export PYTHONPATH=$PYTHONPATH:$(pwd)
export POLARS_SKIP_CPU_CHECK=1

module purge
module load ffmpeg llvm

export LD_LIBRARY_PATH=${FFMPEG_LIB:-}:$LD_LIBRARY_PATH

echo ""
PYTHONUNBUFFERED=1 uv run python3 -m pytest tests/ -v -rA --tb=short --durations=10 "$@"

EXIT_CODE=$?
echo ""
echo "Completed: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
