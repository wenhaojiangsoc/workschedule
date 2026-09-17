#!/bin/bash
#SBATCH --job-name=qwen-ml-eval
#SBATCH --partition=scavenger-gpu
#SBATCH --account=dctrl-as1676
#SBATCH --array=0-4
#SBATCH --gres=gpu:6000_ada_generation:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/hpc/dctrl/as1676/projects/corporate-control/code/SBATCH/logs/ml_eval_%a.out
#SBATCH --error=/hpc/dctrl/as1676/projects/corporate-control/code/SBATCH/logs/ml_eval_%a.err
#SBATCH --mail-user=as1676@duke.edu
#SBATCH --mail-type=BEGIN,FAIL,END

# One array task per outer fold: model outer<K> scores the fold it never trained on.
# Run only after 11.finetune_multilabel.sh has produced all five adapters, then pool with
#   python code/15.aggregate_multilabel_cv.py     (CPU-only, fine on a login node)

cd ~/dctrl_as1676/projects/corporate-control
source /hpc/dctrl/as1676/miniconda3/etc/profile.d/conda.sh
conda activate qwen-ft
export HF_HOME=/hpc/dctrl/as1676/models/hf_cache

export ML_OUTER_FOLD=${SLURM_ARRAY_TASK_ID}
echo "Evaluating outer fold ${ML_OUTER_FOLD}"

python code/12.evaluate_multilabel.py
