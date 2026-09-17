#!/bin/bash
#SBATCH --job-name=ml-stage1
#SBATCH --partition=scavenger-h200
#SBATCH --account=scavenger-h200
#SBATCH --array=0-4%2
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --output=/hpc/group/wenhaolab/projects/corporate-control/code/SBATCH/logs/s1_%A_%a.out
#SBATCH --error=/hpc/group/wenhaolab/projects/corporate-control/code/SBATCH/logs/s1_%A_%a.err
#SBATCH --requeue

# Stage 1: the schedule/job-control gate. One array task per outer fold.
#
# Changed from AS's version, all environmental or throughput, never the recipe's
# statistics:
#   partition h200ea -> scavenger-h200, account -> scavenger-h200 (what this login owns)
#   one 5-GPU job -> a 5-task array of 1-GPU jobs. scavenger-h200 had 253 jobs pending
#     against 30 running with 7 of 9 nodes draining, so a 5-GPU whole-node request would
#     sit in the queue far longer than five single-GPU tasks. Each fold already keeps its
#     own Optuna sqlite study and its own artifact paths, so the folds were never coupled.
#   paths/conda/HF_HOME -> the wenhaolab group space, where the 87 GB model is cached.
# Preemption is per task now, and each task resumes its own study, so losing one fold no
# longer takes the other four with it.

set -u
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

echo "=== stage1 fold ${SLURM_ARRAY_TASK_ID}, job ${SLURM_JOB_ID} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python code/11.finetune_multilabel.py
