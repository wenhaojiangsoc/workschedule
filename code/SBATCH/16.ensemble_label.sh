#!/bin/bash
#SBATCH --job-name=qwen-ml-ensemble
#SBATCH --partition=scavenger-gpu
#SBATCH --account=wenhaolab
#SBATCH --gres=gpu:6000_ada_generation:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --output=/hpc/group/wenhaolab/projects/corporate-control/code/SBATCH/logs/ml_ensemble.out
#SBATCH --error=/hpc/group/wenhaolab/projects/corporate-control/code/SBATCH/logs/ml_ensemble.err
#SBATCH --mail-user=wj93@duke.edu
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --requeue

# Five-model majority-vote labelling of the pre-screened Glassdoor rows.
#
# RUN THE PROBE FIRST. 09.label_glassdoor.py needed multi-day jobs for a SINGLE pass, and
# this is ~3.2 passes (early exit) over the 3.55M rows the binary classifier flagged. Set
# ML_ENSEMBLE_SAMPLE to measure throughput and read the rows/s and mean-models-per-row
# from the log before committing to the full corpus:
#
#     ML_ENSEMBLE_SAMPLE=5000 sbatch code/SBATCH/16.ensemble_label.sh
#
# Then launch the full run with ML_ENSEMBLE_SAMPLE unset. The output is checkpointed and
# id-keyed, so a preempted job resumes where it stopped rather than restarting.

cd /hpc/group/wenhaolab/projects/corporate-control
source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate qwen-ft
export HF_HOME=/hpc/group/wenhaolab/.cache/huggingface

if [ -n "${ML_ENSEMBLE_SAMPLE}" ]; then
    echo "THROUGHPUT PROBE: ${ML_ENSEMBLE_SAMPLE} rows"
    python code/16.ensemble_label.py \
        --sample "${ML_ENSEMBLE_SAMPLE}" \
        --out data/ensemble_throughput_probe.csv
else
    python code/16.ensemble_label.py
fi
