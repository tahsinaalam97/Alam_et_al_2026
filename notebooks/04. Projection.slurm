#!/bin/bash
#SBATCH -p node
#SBATCH --array=0-1000%10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=6-00:00:00
#SBATCH --job-name=Projection
#SBATCH --output=/data/keeling/a/tahsina2/Alam_et_al_2026/slurm/logs/Proj_%A_%a.out
#SBATCH --error=/data/keeling/a/tahsina2/Alam_et_al_2026/slurm/logs/Proj_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tahsina2@illinois.edu

source /data/keeling/a/tahsina2/miniconda3/bin/activate
conda activate pyWBM

mkdir -p /data/keeling/a/tahsina2/Alam_et_al_2026/slurm/logs

PY_SCRIPT="/data/keeling/a/tahsina2/Alam_et_al_2026/slurm/Projection.py"

if [ -z "$TASK_OFFSET" ]; then
    TASK_OFFSET=0
fi

GLOBAL_TASK_ID=$((TASK_OFFSET + SLURM_ARRAY_TASK_ID))

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "TASK_OFFSET: ${TASK_OFFSET}"
echo "GLOBAL_TASK_ID: ${GLOBAL_TASK_ID}"
echo "Date: $(date)"

python "$PY_SCRIPT" "${GLOBAL_TASK_ID}"