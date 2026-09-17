#!/bin/bash
#SBATCH --job-name=ml-stage2
#SBATCH --partition=scavenger-h200
#SBATCH --account=scavenger-h200
#SBATCH --array=0-4%2
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --output=/hpc/group/wenhaolab/projects/corporate-control/code/SBATCH/logs/s2_%x_%A_%a.out
#SBATCH --error=/hpc/group/wenhaolab/projects/corporate-control/code/SBATCH/logs/s2_%x_%A_%a.err
#SBATCH --requeue

# Stage 2: which dimensions, given the branch. ML_BRANCH must be exported at submit time:
#   sbatch --export=ALL,ML_BRANCH=sc --job-name=s2sc code/SBATCH/21.finetune_hierarchical.sh
#   sbatch --export=ALL,ML_BRANCH=jc --job-name=s2jc code/SBATCH/21.finetune_hierarchical.sh
# Folds are inherited from stage 1, so a review held out by the gate is held out here too.

set -u
: "${ML_BRANCH:?ML_BRANCH must be set to sc or jc via --export}"
cd /hpc/group/wenhaolab/projects/corporate-control
source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate qwen-ft
export HF_HOME=/hpc/group/wenhaolab/.cache/huggingface
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export ML_OUTER_FOLD=${SLURM_ARRAY_TASK_ID}
export ML_EVAL_BS=8
# Account caps this login at 2 concurrent GPUs and 24h per job, so the inner search is
# 4 trials x 3 folds instead of 5 x 5. The outer 5-fold CV, which is what the reported
# metric rests on, is untouched; only the hyperparameter search inside each fold is smaller.
export ML_N_TRIALS=4
export ML_N_FOLDS=3

echo "=== stage2 branch ${ML_BRANCH} fold ${SLURM_ARRAY_TASK_ID}, job ${SLURM_JOB_ID} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python code/21.finetune_hierarchical.py
