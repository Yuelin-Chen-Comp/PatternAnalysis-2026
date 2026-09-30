#!/bin/bash
#SBATCH --job-name=hipmri-audit
#SBATCH --partition=cpu
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=hipmri_audit_%j.out
#SBATCH --error=hipmri_audit_%j.err

# Data audit needs no GPU, so it runs on a CPU node (keeps the A100s free).
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"

source $HOME/miniconda3/bin/activate
conda activate torch

python -c "import nibabel" 2>/dev/null || pip install --no-cache-dir nibabel

# sbatch runs a spooled copy of this script, so locate data_audit.py via the submit dir
python "$SLURM_SUBMIT_DIR/data_audit.py" --json-out "$HOME/hipmri_audit_summary_${SLURM_JOB_ID}.json"
