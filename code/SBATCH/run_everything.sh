#!/bin/bash
#SBATCH --job-name=cc
#SBATCH --partition=scavenger-gpu
#SBATCH --account=wenhaolab
#SBATCH --array=0-17
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=3-00:00:00
#SBATCH --output=/hpc/group/wenhaolab/projects/corporate-control/code/SBATCH/logs/cc_%A_%a.out
#SBATCH --error=/hpc/group/wenhaolab/projects/corporate-control/code/SBATCH/logs/cc_%A_%a.err
#SBATCH --requeue
# Node exclusions live in .exclude_nodes / .bad_nodes and are applied at submit time,
# not baked in here, so a node discovered to be faulty is avoided by every later resubmit.

# EVERYTHING, in one submission, on scavenger-gpu under the wenhaolab account, which has
# no GPU cap and a 7-day wall limit. scavenger-h200 is faster per card but caps this login
# at 2 concurrent GPUs and 24h, which serialised the work.
#
#   tasks  0-4   stage 1, the schedule/job-control gate, outer folds 0-4
#   tasks  5-9   stage 2 branch sc, 20 schedule dimensions, outer folds 0-4
#   tasks 10-14  stage 2 branch jc, 7 job-control dimensions, outer folds 0-4
#   tasks 15-17  saturation learning curves for s1, sc, jc
#
# --exclude drops every 2080 node (11GB) and every a5000 node (24GB): two cards must hold
# a 4-bit 35B model plus ~8GB of logits, so 32GB per card is the floor. That leaves 26
# eligible GPUs, so about 13 of these 18 tasks run at once and the rest queue behind them.
#
# Self-continuing: Optuna keeps a per-task sqlite study, training resumes from the last
# HF checkpoint, the saturation curve skips points already in its JSON, and each task
# resubmits itself until it writes a DONE marker. One sbatch is the whole protocol.

set -u
PROJ=/hpc/group/wenhaolab/projects/corporate-control

cd "$PROJ"
source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate qwen-ft
export HF_HOME=/hpc/group/wenhaolab/.cache/huggingface
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# Unsloth writes generated source to $PWD/unsloth_compiled_cache by default. Every task
# shares this working directory and the training scripts rmtree that cache before each
# model load, so concurrent tasks raced and the loser died with
# "OSError: could not get source code". Give each task its own cache location.
export UNSLOTH_COMPILE_LOCATION="$PROJ/.unsloth_cache/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$UNSLOTH_COMPILE_LOCATION"
# Effective batch stays 8. Two 32GB cards are the worst case that survives --exclude.
export ML_N_TRIALS=4 ML_N_FOLDS=3

T=${SLURM_ARRAY_TASK_ID}
MAX_ATTEMPTS=24
SLUG=Qwen3.5-35B-A3B

if   [ "$T" -lt 5  ]; then KIND=s1;  FOLD=$T;        RES="models/cv_results_multilabel/${SLUG}/outer${FOLD}"
elif [ "$T" -lt 10 ]; then KIND=sc;  FOLD=$((T-5));  RES="models/cv_results_hier/${SLUG}/sc/outer${FOLD}"
elif [ "$T" -lt 15 ]; then KIND=jc;  FOLD=$((T-10)); RES="models/cv_results_hier/${SLUG}/jc/outer${FOLD}"
else SATS=(s1 sc jc); KIND=sat; SAT=${SATS[$((T-15))]}; RES="models/saturation/${SLUG}/${SAT}"
fi

mkdir -p "$RES"

# Some nodes accept the job and report a GPU in the prolog, but torch cannot open it
# ("Unsloth cannot find any torch accelerator", "Failed to get device handle for GPU 0").
# One such node ate 191 launches and exhausted the retry budget of eight tasks, because
# every resubmit landed straight back on the only free node. Check the GPU is real BEFORE
# counting an attempt; if it is not, blacklist the node and requeue elsewhere for free.
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count()>0 else 1)" 2>/dev/null; then
    echo "GPU UNUSABLE on $(hostname): blacklisting the node and requeueing (attempt NOT counted)."
    echo "$(hostname)" >> "$PROJ/.bad_nodes"
    EXC=$(cat "$PROJ/.exclude_nodes" "$PROJ/.bad_nodes" 2>/dev/null | sort -u | paste -sd,)
    sbatch --array=${T} --partition="${SLURM_JOB_PARTITION}" --account="${SLURM_JOB_ACCOUNT}" \
           --exclude="${EXC}" "$PROJ/code/SBATCH/run_everything.sh"
    exit 0
fi

ATT=$(cat "${RES}/.attempts" 2>/dev/null || echo 0); ATT=$((ATT+1)); echo "$ATT" > "${RES}/.attempts"
echo "=== task ${T}: ${KIND}${SAT:-} ${FOLD:-} attempt ${ATT}/${MAX_ATTEMPTS} job ${SLURM_JOB_ID} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Effective batch is ALWAYS 8. How it is split depends on the card and the branch:
# stage 2 prompts are ~2000 tokens against ~1150 for stage 1, and the per-step logits
# tensor (batch x seq x 248046 vocab, upcast to fp32) is what actually blows up.
GPUMEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$KIND" = "s1" ]; then BS=8; EB=4; else BS=4; EB=2; fi
if [ "${GPUMEM:-0}" -lt 40000 ]; then BS=$((BS / 2)); EB=1; fi
export ML_TRAIN_BS=$BS ML_GRAD_ACCUM=$((8 / BS)) ML_EVAL_BS=$EB
echo "batch sizing: gpu_mem=${GPUMEM}MiB kind=${KIND} -> train_bs=${BS} accum=$((8 / BS)) eval_bs=${EB}"
[ -f "${RES}/DONE" ] && { echo "already DONE"; exit 0; }

case "$KIND" in
  s1) ML_OUTER_FOLD=${FOLD} python code/11.finetune_multilabel.py ;;
  sc|jc) ML_BRANCH=${KIND} ML_OUTER_FOLD=${FOLD} python code/21.finetune_hierarchical.py ;;
  sat) ML_SAT_TASK=${SAT} ML_SAT_SEEDS=0,1 python code/30.saturation.py ;;
esac
rc=$?

[ -f "${RES}/DONE" ] && { echo "task ${T} finished (rc=${rc})"; exit 0; }
if [ "$ATT" -ge "$MAX_ATTEMPTS" ]; then
  echo "task ${T} hit MAX_ATTEMPTS (rc=${rc}); not resubmitting."; exit 1
fi
echo "task ${T} did not finish (rc=${rc}); resubmitting this task."
EXC=$(cat "$PROJ/.exclude_nodes" "$PROJ/.bad_nodes" 2>/dev/null | sort -u | paste -sd,)
sbatch --array=${T} --partition="${SLURM_JOB_PARTITION}" --account="${SLURM_JOB_ACCOUNT}" \
       --exclude="${EXC}" "$PROJ/code/SBATCH/run_everything.sh"
exit 0
