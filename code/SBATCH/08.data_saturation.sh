#!/bin/bash
#SBATCH --job-name=qwen35b-saturation
#SBATCH --partition=scavenger-gpu
#SBATCH --account=wenhaolab
#SBATCH --gres=gpu:6000_ada_generation:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=/hpc/group/wenhaolab/projects/corporate-control/code/SBATCH/logs/saturation.out
#SBATCH --error=/hpc/group/wenhaolab/projects/corporate-control/code/SBATCH/logs/saturation.err
#SBATCH --mail-user=wj93@duke.edu
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --requeue


# Setup
cd /hpc/group/wenhaolab/projects/corporate-control

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate qwen-ft

export HF_HOME=/hpc/group/wenhaolab/.cache/huggingface

pip install -q matplotlib

python code/08.data_saturation.py
