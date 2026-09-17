#!/bin/bash
#SBATCH --job-name=qwen35b-ml-smoke
#SBATCH --partition=scavenger-h200
#SBATCH --account=scavenger-h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/hpc/group/wenhaolab/projects/corporate-control/code/SBATCH/logs/ml_smoke_%j.out
#SBATCH --error=/hpc/group/wenhaolab/projects/corporate-control/code/SBATCH/logs/ml_smoke_%j.err
#SBATCH --requeue

# Adapted from the original by AS for the wenhaolab account on Duke DCC.
# Changes from that version, all environmental:
#   partition h200ea -> scavenger-h200 (h200ea is not visible to this account;
#                       scavenger-h200 has 9 nodes of 8x H200, 192 cores, 2 TB)
#   account wenhaolab -> scavenger-h200 (the only account this partition accepts)
#   --nodes=1 added so the 5 GPUs land on ONE node (the loop pins one GPU per fold
#                       with CUDA_VISIBLE_DEVICES, which only works within a node)
#   mem 640G -> 500G   (comfortably under the node's 2 TB and faster to schedule)
#   paths -> /hpc/group/wenhaolab/projects/corporate-control
#   conda -> cluster Anaconda + the qwen-ft env cloned from socialvlm
#   HF_HOME -> the group cache that already holds Qwen3.5-35B-A3B
# The training recipe itself is untouched.

set -u
cd /hpc/group/wenhaolab/projects/corporate-control

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate qwen-ft

# Model is already cached here (87 GB, 14 shards). Compute nodes have no internet,
# so this must point at a warm cache or the job dies at model load.
export HF_HOME=/hpc/group/wenhaolab/.cache/huggingface
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

if [ ! -f data/trainingfinal/labelled_cv.csv ]; then
    echo "FATAL: data/trainingfinal/labelled_cv.csv is missing."
    echo "  Build it once on the login node:"
    echo "    python code/10.prepare_multilabel_data.py --encoding utf-8"
    exit 1
fi

echo "=== job ${SLURM_JOB_ID} on $(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python -c "import pandas as pd; d=pd.read_csv('data/trainingfinal/labelled_cv.csv'); print('data:', len(d), 'rows; folds', sorted(d.fold.unique()), '; states', d.strat.value_counts().sort_index().to_dict())"

export ML_OUTER_FOLD=0
export ML_N_TRIALS=1
export ML_N_FOLDS=2
export ML_EPOCHS=1
export ML_STUDY_TAG=smoke
export ML_SKIP_FINAL=1
export CUDA_VISIBLE_DEVICES=0

echo "=== SMOKE: 1 trial x 2 inner folds x 1 epoch, outer fold 0, no final training ==="
python code/11.finetune_multilabel.py
