#!/bin/bash
#SBATCH --job-name=hipmri-orient
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --time=00:15:00
#SBATCH --output=hipmri_orient_%j.out
#SBATCH --error=hipmri_orient_%j.err

# Reads about 40 label volumes; needs no GPU, but the 'cpu' partition rejected our memory
# request on Rangpur, so we reuse the course partition/account (same as audit_job.sh).
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"

source $HOME/miniconda3/bin/activate
conda activate torch

# sbatch runs a spooled copy of this script, so locate the .py via the submit dir
python "$SLURM_SUBMIT_DIR/check_orientation.py" --n 40
