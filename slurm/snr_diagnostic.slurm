#!/bin/bash
#SBATCH --job-name=snr_diag
#SBATCH --output=logs/tests/snr_diag_%j.out
#SBATCH --error=logs/tests/snr_diag_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# Set your partition here, or export SBATCH_PARTITION (it overrides this line).
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:1

DATASET="${1:-seedlings_1}"

echo "SNR VAD Masking Diagnostic"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)"
echo "Dataset: $DATASET"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"

export LD_LIBRARY_PATH=${FFMPEG_LIB:-}:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
export POLARS_SKIP_CPU_CHECK=1

module purge
module load ffmpeg

echo ""
mkdir -p logs/tests

PYTHONUNBUFFERED=1 \
uv run python -m src.analysis.snr_vad_diagnostic "$DATASET"

EXIT_CODE=$?

echo ""
echo "Completed: $(date '+%H:%M:%S')  exit=$EXIT_CODE"
exit $EXIT_CODE
