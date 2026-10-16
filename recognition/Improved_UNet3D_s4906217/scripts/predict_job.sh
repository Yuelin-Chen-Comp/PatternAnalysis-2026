#!/bin/bash
#SBATCH --job-name=hipmri-predict
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=hipmri_predict_%j.out
#SBATCH --error=hipmri_predict_%j.err

# Run a trained model on a split and analyse where it fails (see predict.py): per-class Dice and IoU, Dice of the
# apex / middle / base third of the prostate, leakage into the rectum and bladder, and figures of the worst cases.
# The results go to ~/hipmri_runs/<run-name>/predict_<split>/. Submit from the scripts/ folder:
#     cd scripts && sbatch predict_job.sh baseline2d                 # validation split (development)
#     cd scripts && sbatch predict_job.sh unet3d --split test        # the final analysis on the test patients
# Everything after the run name goes to predict.py unchanged.
NAME=${1:?usage: sbatch predict_job.sh <run-name> [predict.py arguments]}
shift 1

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

# name the environment explicitly, otherwise predict.py's arguments would be passed on to conda
source $HOME/miniconda3/bin/activate torch

python -c "import nibabel" 2>/dev/null || pip install --no-cache-dir nibabel
python -c "import matplotlib" 2>/dev/null || pip install --no-cache-dir matplotlib
python -c "import scipy" 2>/dev/null || pip install --no-cache-dir scipy

# sbatch runs a spooled copy of this script, so locate predict.py via the submit dir (scripts/)
python "$SLURM_SUBMIT_DIR/../predict.py" --name "$NAME" ${CACHE_DIR:+--cache-dir "$CACHE_DIR"} "$@"
