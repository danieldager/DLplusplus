#!/bin/bash
#SBATCH --job-name=vad_ten
#SBATCH --output=logs/vad/ten_%j.out
#SBATCH --error=logs/vad/ten_%j.err
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=24:00:00
# Set your partition here, or export SBATCH_PARTITION (it overrides this line).
#SBATCH --partition=<partition>

DATASET="$1"
shift 1  # remaining args (--manifest, --path-col, --audio-root) passed to vad.py

if [ -z "$DATASET" ]; then
    echo "Usage: sbatch vad.slurm <DATASET> [--manifest PATH] [--path-col COL] [--audio-root DIR]"
    echo "Example: sbatch vad.slurm chunks30"
    echo "Example: sbatch vad.slurm my_data --manifest /data/meta.xlsx --path-col recording_id"
    exit 1
fi

echo ""
echo "Step 1/3  TenVAD"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Job: $SLURM_JOB_ID  Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK  Mem: ${SLURM_MEM_PER_NODE}"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"

export LD_LIBRARY_PATH=${FFMPEG_LIB:-}:$LD_LIBRARY_PATH
export POLARS_SKIP_CPU_CHECK=1

module purge
module load ffmpeg llvm

export PYTHONPATH=$PYTHONPATH:$(pwd)

PYTHONUNBUFFERED=1 \
uv run src/pipeline/vad.py "$DATASET" \
    --workers "$SLURM_CPUS_PER_TASK" \
    "$@"

EXIT_CODE=$?

echo ""
echo "VAD completed: $(date '+%H:%M:%S')  exit=$EXIT_CODE"
exit $EXIT_CODE
