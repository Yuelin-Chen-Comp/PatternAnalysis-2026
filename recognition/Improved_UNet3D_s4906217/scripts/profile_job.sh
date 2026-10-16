#!/bin/bash
#SBATCH --job-name=hipmri-prof
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=hipmri_prof_%j.out
#SBATCH --error=hipmri_prof_%j.err

# Where does the time of one validation volume go? See profile_eval.py. On the real validation volumes with the
# trained 2D baseline (~/hipmri_runs/baseline2d/best.pt), plain and after a training pass with data-loader workers.
# Submit from the scripts/ folder (only one of your jobs runs at a time, so wait for a running training job):
#     cd scripts && sbatch profile_job.sh
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

source $HOME/miniconda3/bin/activate torch
python -c "import nibabel" 2>/dev/null || pip install --no-cache-dir nibabel

# sbatch runs a spooled copy of this script, so locate the .py via the submit dir
python "$SLURM_SUBMIT_DIR/profile_eval.py" --model unet2d --real 6 \
    --checkpoint "$HOME/hipmri_runs/baseline2d/best.pt" --like-training
