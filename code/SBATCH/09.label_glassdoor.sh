#!/bin/bash
#SBATCH --job-name=glassdoor-label
#SBATCH --partition=scavenger-gpu
#SBATCH --account=wenhaolab
#SBATCH --gres=gpu:6000_ada_generation:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=4-00:00:00
#SBATCH --output=/hpc/group/wenhaolab/projects/corporate-control/code/SBATCH/logs/glassdoor_label.out
#SBATCH --error=/hpc/group/wenhaolab/projects/corporate-control/code/SBATCH/logs/glassdoor_label.err
#SBATCH --mail-user=wj93@duke.edu
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --requeue

cd /hpc/group/wenhaolab/projects/corporate-control
source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate qwen-ft
export HF_HOME=/hpc/group/wenhaolab/.cache/huggingface

python code/09.label_glassdoor.py
