#!/bin/bash
#SBATCH --job-name=hipmri-audit
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --time=00:20:00
#SBATCH --output=hipmri_audit_%j.out
#SBATCH --error=hipmri_audit_%j.err

# The audit itself needs no GPU, but the 'cpu' partition rejected our memory request
# on Rangpur, so we reuse the course partition/account that worked in Demo 2. It is short.
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"

source $HOME/miniconda3/bin/activate
conda activate torch

python -c "import nibabel" 2>/dev/null || pip install --no-cache-dir nibabel

# sbatch runs a spooled copy of this script, so locate data_audit.py via the submit dir
python "$SLURM_SUBMIT_DIR/data_audit.py" --json-out "$HOME/hipmri_audit_summary_${SLURM_JOB_ID}.json"
