#!/bin/bash
#SBATCH --job-name=vtc_clips
#SBATCH --output=logs/vtc/vtc_clips_%A_%a.out
#SBATCH --error=logs/vtc/vtc_clips_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# Set your partition here, or export SBATCH_PARTITION (it overrides this line).
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:1
#SBATCH --array=0-5

DATASET="$1"
shift 1  # remaining args passed through to vtc_on_clips.py

if [ -z "$DATASET" ]; then
    echo "Usage: sbatch slurm/vtc_clips.slurm <DATASET> [ADDITIONAL_ARGS...]"
    echo "Example: sbatch slurm/vtc_clips.slurm seedlings_10"
    exit 1
fi

echo ""
echo "VTC on Clips"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  Node: $(hostname)"
echo "Shard: ${SLURM_ARRAY_TASK_ID}/$((SLURM_ARRAY_TASK_COUNT-1))"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"

export LD_LIBRARY_PATH=${FFMPEG_LIB:-}:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
export POLARS_SKIP_CPU_CHECK=1

module purge
module load ffmpeg
module load git-lfs

echo ""

PYTHONUNBUFFERED=1 \
uv run src/analysis/vtc_on_clips.py "$DATASET" \
    --array_id "$SLURM_ARRAY_TASK_ID" \
    --array_count "$SLURM_ARRAY_TASK_COUNT" \
    "$@"

EXIT_CODE=$?

echo ""
echo "Completed: $(date '+%H:%M:%S')  exit=$EXIT_CODE"
exit $EXIT_CODE
