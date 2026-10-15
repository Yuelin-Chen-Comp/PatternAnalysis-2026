#!/bin/bash
#SBATCH --job-name=hipmri-vram
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --time=00:30:00
#SBATCH --output=hipmri_vram_%j.out
#SBATCH --error=hipmri_vram_%j.err

# Peak GPU memory of the 2D and 3D U-Net on random tensors (no dataset needed); see measure_vram.py.
# One GPU, no --mem (the comp3710 partition rejects it). Submit from the scripts/ folder:
#     cd scripts && sbatch vram_job.sh
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

source $HOME/miniconda3/bin/activate
conda activate torch

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# sbatch runs a spooled copy of this script, so locate the .py via the submit dir
python "$SLURM_SUBMIT_DIR/measure_vram.py" --json-out "$HOME/hipmri_vram_${SLURM_JOB_ID}.json"
