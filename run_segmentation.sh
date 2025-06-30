#!/bin/bash
#SBATCH --job-name=eeg_seg              # minimal job name
#SBATCH --partition=cpu_medium          # Big Purple CPU queue
#SBATCH --time=00:02:00                  # wall-clock limit (hh:mm:ss)
#SBATCH --cpus-per-task=1              # threads for Python / MNE
#SBATCH --mem=64G                      # RAM for the job
#SBATCH --chdir=/gpfs/scratch/np3106/eeg_cleaning   # work dir = input/output root
#SBATCH --output=slurm-%j.out           # stdout & stderr → slurm-<jobid>.out

# ---- load your Python environment (edit to match your setup) ----
module purge
module load anaconda3/cpu/2022.10            # or any module that provides Conda
conda activate /gpfs/scratch/np3106/conda-envs/eeg-clean                # env with mne, numpy, etc.

# ---- run the script ------------------------------------------------
python data_segmentation.py