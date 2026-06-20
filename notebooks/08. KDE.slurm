#!/bin/bash
##================ SLURM Job Submission Script ===================
#SBATCH -p node
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --job-name=KDE
#SBATCH --output=/data/keeling/a/tahsina2/Alam_et_al_2026/slurm/logs/KDE_%j.out
#SBATCH --error=/data/keeling/a/tahsina2/Alam_et_al_2026/slurm/logs/KDE_%j.err

##================ EMAIL NOTIFICATIONS ===================
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tahsina2@illinois.edu

set -euo pipefail

##================ ENVIRONMENT SETUP ===================
mkdir -p /data/keeling/a/tahsina2/Alam_et_al_2026/slurm/logs
mkdir -p /data/keeling/a/tahsina2/Alam_et_al_2026/slurm/job_reports

source /data/keeling/a/tahsina2/miniconda3/bin/activate
conda activate pyWBM

PY_SCRIPT="/data/keeling/a/tahsina2/Alam_et_al_2026/slurm/KDE.py"

echo "#####################################################"
echo "#  Starting KDE run"
echo "#  Date: $(date)"
echo "#  JobID: ${SLURM_JOB_ID}"
echo "#  Host: $(hostname)"
echo "#####################################################"

##================ RUN PYTHON SCRIPT ===================
python "$PY_SCRIPT"

STATUS=$?

##================ RESOURCE USAGE SUMMARY ===================
echo ""
echo "#####################################################"
echo "#  Job Completed"
echo "#  Date: $(date)"
echo "#  Exit status: ${STATUS}"
echo "#  Collecting job resource usage:"
echo "#####################################################"

RUNSTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="/data/keeling/a/tahsina2/Alam_et_al_2026/slurm/job_reports"
REPORT_FILE="${REPORT_DIR}/resource_usage_${SLURM_JOB_ID}_${RUNSTAMP}.log"

{
  echo "===== Resource Usage Report ====="
  echo "JobID:        ${SLURM_JOB_ID}"
  echo "JobName:      ${SLURM_JOB_NAME}"
  echo "Report time:  $(date)"
  echo ""
  echo "---- sacct summary (job + steps) ----"
  sacct -j "${SLURM_JOB_ID}" \
    --format=JobID,JobName%30,Partition,State,ExitCode,Elapsed,AllocCPUS,ReqMem,MaxRSS,MaxVMSize%20,AveRSS,AveCPU
} | tee "$REPORT_FILE"

echo "#####################################################"
echo "#  Resource report saved to: $REPORT_FILE"
echo "#####################################################"

exit ${STATUS}