#!/bin/bash
#SBATCH --job-name=qwen-eval
#SBATCH --partition=scavenger-gpu
#SBATCH --account=wenhaolab
#SBATCH --gres=gpu:6000_ada_generation:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/hpc/group/wenhaolab/projects/corporate-control/code/SBATCH/logs/eval.out
#SBATCH --error=/hpc/group/wenhaolab/projects/corporate-control/code/SBATCH/logs/eval.err
#SBATCH --mail-user=wj93@duke.edu
#SBATCH --mail-type=BEGIN,FAIL,END

cd /hpc/group/wenhaolab/projects/corporate-control
source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate qwen-ft
export HF_HOME=/hpc/group/wenhaolab/.cache/huggingface

python code/07.evaluate_qwen.py
