#!/bin/bash
#SBATCH -p node
#SBATCH --array=0-679%10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=Calibration

#SBATCH --output=/data/keeling/a/tahsina2/Alam_et_al_2026/slurm/logs/cal_%A_%a.out
#SBATCH --error=/data/keeling/a/tahsina2/Alam_et_al_2026/slurm/logs/cal_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tahsina2@illinois.edu

set -euo pipefail

source /data/keeling/a/tahsina2/miniconda3/bin/activate
conda activate pyWBM

mkdir -p /data/keeling/a/tahsina2/Alam_et_al_2026/slurm/logs

echo "#####################################################"
echo "#  Starting single-scenario calibration"
echo "#  Date: $(date)"
echo "#  JobID: ${SLURM_JOB_ID}"
echo "#  ArrayTaskID: ${SLURM_ARRAY_TASK_ID}"
echo "#####################################################"

python /data/keeling/a/tahsina2/Alam_et_al_2026/slurm/Calibration.py "${SLURM_ARRAY_TASK_ID}"

echo ""
echo "#####################################################"
echo "#  Job Completed"
echo "#  Date: $(date)"
echo "#  Collecting job resource usage:"
echo "#####################################################"

RUNSTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="${SLURM_SUBMIT_DIR:-$PWD}/job_reports"
mkdir -p "$REPORT_DIR"
REPORT_FILE="${REPORT_DIR}/resource_usage_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${RUNSTAMP}.log"

{
  echo "===== Resource Usage Report ====="
  echo "JobID:        ${SLURM_JOB_ID}"
  echo "ArrayTaskID:  ${SLURM_ARRAY_TASK_ID}"
  echo "JobName:      ${SLURM_JOB_NAME}"
  echo "Report time:  $(date)"
  echo ""
  echo "---- sacct summary (job + steps) ----"
  sacct -j "${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    --format=JobID,JobName%30,Partition,State,ExitCode,Elapsed,AllocCPUS,ReqMem,MaxRSS,MaxVMSize%20,AveRSS,AveCPU
} | tee "$REPORT_FILE"

echo "#####################################################"
echo "#  Resource report saved to: $REPORT_FILE"
echo "#####################################################"