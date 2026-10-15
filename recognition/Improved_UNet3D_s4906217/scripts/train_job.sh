#!/bin/bash
#SBATCH --job-name=hipmri-train
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=hipmri_train_%j.out
#SBATCH --error=hipmri_train_%j.err

# Train the 2D baseline or the 3D U-Net (see train.py). One GPU, no --mem (the comp3710 partition rejects it).
# Submit from the scripts/ folder:
#     cd scripts && sbatch train_job.sh unet2d baseline2d
#     cd scripts && sbatch train_job.sh unet3d unet3d --epochs 150
# Everything after the run name goes to train.py unchanged. Add --test to score the test split after training.
#
# Optional: CACHE_DIR=$HOME/hipmri_cache sbatch --export=ALL train_job.sh unet2d baseline2d
# keeps the preprocessed volumes (about 25 MB each, about 5 GB for all 211: mind the home quota). Without it
# every run reads and normalises the NIfTI files again, which takes a few minutes.
MODEL=${1:?usage: sbatch train_job.sh <unet2d|unet3d> <run-name> [train.py arguments]}
NAME=${2:?usage: sbatch train_job.sh <unet2d|unet3d> <run-name> [train.py arguments]}
shift 2

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

source $HOME/miniconda3/bin/activate
conda activate torch

python -c "import nibabel" 2>/dev/null || pip install --no-cache-dir nibabel
python -c "import matplotlib" 2>/dev/null || pip install --no-cache-dir matplotlib

# sbatch runs a spooled copy of this script, so locate train.py via the submit dir (scripts/)
python "$SLURM_SUBMIT_DIR/../train.py" --model "$MODEL" --name "$NAME" \
    --workers "${SLURM_CPUS_PER_TASK:-1}" ${CACHE_DIR:+--cache-dir "$CACHE_DIR"} "$@"
