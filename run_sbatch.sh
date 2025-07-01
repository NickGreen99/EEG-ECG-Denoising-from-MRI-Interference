#!/bin/bash
#SBATCH --job-name=eeg_seg              # minimal job name
#SBATCH --partition=gpu4_short          # Big Purple CPU queue
#SBATCH --time=04:00:00                  # wall-clock limit (hh:mm:ss)
#SBATCH --cpus-per-task=4              # threads for Python / MNE
#SBATCH --mem=64G                      # RAM for the job
#SBATCH --gres=gpu:1  
#SBATCH --chdir=/gpfs/scratch/np3106/eeg_cleaning   # work dir = input/output root
#SBATCH --output=slurm-%j.out           # stdout & stderr → slurm-<jobid>.out

# ---- load your Python environment (edit to match your setup) ----
module purge
module load anaconda3/gpu/new            # or any module that provides Conda

# 🧠 Directly activate custom environment
export PATH=/gpfs/scratch/np3106/venvs/eeg-clean/bin:$PATH

which python

# ---- run the script ------------------------------------------------
python train.py