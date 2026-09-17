"""
Fine-tune Qwen3.5-35B-A3B (LoRA, Unsloth) for the two-dimension classifier:
each review gets schedule_related and job_control_related in {0,1}, emitted as two
digits "<schedule><jobctrl>".

Adapted from 06.finetune.py (binary schedule classifier). Differences:
  - Target is two digits instead of one; loss covers both.
  - Cross-validation is stratified on the 4-way state key (2*schedule + job_control),
    so plain StratifiedKFold stays multi-label-aware without a new dependency.
  - Optuna objective is the *schedule* F1 (the primary/headline dimension), with mean
    macro-F1 kept as the tie-breaker (schedule saturates, so the tie-break is what
    actually selects a model that is also good at the rare job_control dimension).
  - Optional oversampling of the rare states (job-control / both / none) in the TRAIN
    split only, to give the minority signal more weight.
  - Resumable Optuna study (sqlite) so a preempted scavenger-gpu job continues its
    search instead of restarting from scratch.

Recipe v2 (2026-07-28) after the v1 run scored 0.0 on every fold:
  - Sequence budget lives in multilabel_prompt.py and is checked by a preflight before
    any GPU work. v1 ran at 512 tokens against a ~1030-token system prompt, so every
    training example was truncated before the answer digits and the model was never
    supervised on the answer at all.
  - Loss is masked to the completion, so eval_loss (and therefore early stopping and
    checkpoint selection) tracks answer accuracy rather than prompt reconstruction.
  - LoRA is attention-only. Including gate/up/down_proj made Unsloth attach LoRA to all
    256 MoE experts, which OOM'd two of five trials.
  - Raw generations are logged and dumped per fold; a mostly-malformed fold aborts the
    study, and a degenerate search refuses to train a final model.

Nested CV (2026-09-02): this script trains ONE outer fold. `ML_OUTER_FOLD=K` selects it;
the model trains on every row with `fold != K` (including the pinned few-shot rows at
fold == -1) and never sees `fold == K`, which 12.evaluate_multilabel.py scores. The whole
Optuna search -- inner folds and all -- happens inside the outer training pool, so no
hyperparameter information crosses into the outer test fold. Every artifact path and the
Optuna study name carry `outer{K}`, so the five folds run as an independent Slurm job array
without colliding on disk or in sqlite.

Run from the repo root: `ML_OUTER_FOLD=0 python code/11.finetune_multilabel.py`
Smoke run: `ML_OUTER_FOLD=0 ML_N_TRIALS=1 ML_N_FOLDS=2 ML_EPOCHS=1 python code/11.finetune_multilabel.py`
"""

import os

# Which outer fold this job holds out. Required -- defaulting it would silently retrain
# fold 0 five times in a Slurm array whose task ids failed to propagate. Checked before
# the heavy imports so a mis-launched array task fails in a second, not after 30s of
# torch/unsloth startup on five GPUs.
if "ML_OUTER_FOLD" not in os.environ:
    raise SystemExit(
        "ML_OUTER_FOLD is not set. This script trains ONE outer fold of the nested CV.\n"
        "  Run a single fold:  ML_OUTER_FOLD=0 python code/11.finetune_multilabel.py\n"
        "  All five folds:     sbatch code/SBATCH/11.finetune_multilabel.sh  (--array=0-4)"
    )
OUTER_FOLD = int(os.environ["ML_OUTER_FOLD"])

import sys
import json
import gc
import shutil
import hashlib
import dataclasses
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import optuna
from optuna.trial import TrialState
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for local modules
from metrics_multilabel import parse_two_digit
from multilabel_prompt import (
    MODEL_NAME, MODEL_SLUG, MAX_SEQ_LEN, SYSTEM_PROMPT, ANSWER_MARGIN,
    format_prompt, format_answer, format_example, generate_two_digit,
    preflight_token_budget,
)

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback

# ============================================================
# Configuration
# ============================================================
# MODEL_NAME / MODEL_SLUG / MAX_SEQ_LEN / SYSTEM_PROMPT and the prompt formatters come
# from multilabel_prompt.py, shared with 12.evaluate_multilabel.py so they cannot drift.
DATA_CSV = "data/trainingfinal/labelled_cv.csv"

RESULTS_DIR = f"models/cv_results_multilabel/{MODEL_SLUG}/outer{OUTER_FOLD}"
FINAL_DIR   = f"models/finetuned/{MODEL_SLUG}_multilabel/outer{OUTER_FOLD}"

# Bump whenever the training recipe changes. Feeds the Optuna study name, the sqlite
# filename, and the fingerprint guard, so a changed recipe can never silently resume
# an old study. v1 was the seq-len-512 run whose folds all scored 0.0; v2 was the single
# 75/25 split on 778 rows; v3 is the outer-5-fold nested CV on the ~1578-row expansion.
RECIPE_VERSION = "v4"

# Env overrides so a smoke run needs no code edit (see SBATCH/11b.smoke_multilabel.sh).
N_FOLDS  = int(os.environ.get("ML_N_FOLDS", 5))
N_TRIALS = int(os.environ.get("ML_N_TRIALS", 5))
N_EPOCHS = float(os.environ.get("ML_EPOCHS", 5))
PATIENCE = 3
# Effective batch stays 8 (TRAIN_BS x GRAD_ACCUM) but runs as one real batch instead of
# eight sequential 1-row microbatches. Sequences are ~1.1k tokens; the H200 has 140GB,
# where the original recipe was tuned for a 47GB card.
TRAIN_BS   = int(os.environ.get("ML_TRAIN_BS", 8))
GRAD_ACCUM = int(os.environ.get("ML_GRAD_ACCUM", 1))
EVAL_BS    = int(os.environ.get("ML_EVAL_BS", 16))

# ML_STUDY_TAG gives a throwaway run (e.g. the smoke test) its own sqlite study so its
# trials can never be counted toward, or selected by, the real search. ML_SKIP_FINAL
# stops it from overwriting the production adapter in FINAL_DIR.
STUDY_TAG  = os.environ.get("ML_STUDY_TAG", "").strip()
SKIP_FINAL = os.environ.get("ML_SKIP_FINAL", "").strip() not in ("", "0")

# Attention-only LoRA. Including gate/up/down_proj makes Unsloth attach LoRA to all 256
# MoE experts too (232.8M trainable params at r=4, 931M at r=16), which is what OOM'd
# trials 3 and 4 of the previous run and leaves no headroom for the real sequence length.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Oversampling of rare states in the TRAIN split only (never val/test).
# Keys are the 4-way state code (2*schedule + job_control): 0=none,1=jobctrl,2=sched,3=both.
# Derived from the outer training pool by compute_oversample_factors() rather than fixed:
# the old flat {0:3, 1:3, 3:3} was tuned for the 626/53/46/53 round-1 distribution, and
# round 2 was mined precisely to fill the job-control states, so a flat 3x would now
# over-weight states that no longer need it.
OVERSAMPLE = True
OVERSAMPLE_CAP = 3       # never replicate a state more than this
OVERSAMPLE_TARGET = 3    # aim each rare state at ~1/3 of the majority state's count
OVERSAMPLE_FACTORS = {}  # filled in at startup from the outer training pool

# A trial whose generations are mostly unparseable is a setup bug, not a bad
# hyperparameter -- abort rather than record it as a legitimate 0.0.
MALFORMED_ABORT_FRAC = 0.25
LOG_FIRST_N_GENERATIONS = 8
MIN_USABLE_F1 = 0.50


class DegenerateOutputError(Exception):
    """Generations unparseable at scale: a setup bug, not a hyperparameter.

    Subclasses Exception directly, NOT RuntimeError, so Optuna's `catch=` tuple
    cannot swallow it into a FAIL trial.
    """


def compute_oversample_factors(pool):
    """Replication factor per rare state, derived from the outer training pool.

    factor = clip(round(n_majority / (OVERSAMPLE_TARGET * n_state)), 1, OVERSAMPLE_CAP),
    so a state already at a third of the majority is left alone and nothing is replicated
    more than OVERSAMPLE_CAP times. Computed from THIS outer fold's training rows only.
    """
    counts = pool["strat"].value_counts().to_dict()
    if not counts:
        return {}
    majority_state = max(counts, key=counts.get)
    n_majority = counts[majority_state]

    factors = {}
    for state, n in sorted(counts.items()):
        if state == majority_state or n == 0:
            continue
        f = int(np.clip(round(n_majority / (OVERSAMPLE_TARGET * n)), 1, OVERSAMPLE_CAP))
        if f > 1:
            factors[state] = f

    print(f"Oversample factors (majority state={majority_state}, n={n_majority}): "
          f"{ {s: (counts[s], f) for s, f in factors.items()} }  [state: (n, factor)]")
    return factors


def maybe_oversample(df):
    """Replicate rare-state rows in a TRAIN dataframe. No-op on val/test."""
    if not OVERSAMPLE:
        return df
    parts = [df]
    for state, factor in OVERSAMPLE_FACTORS.items():
        if factor > 1:
            sub = df[df["strat"] == state]
            parts.extend([sub] * (factor - 1))
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------
# SFTConfig capability detection
#
# TRL renamed `max_seq_length` -> `max_length` and added prompt/completion datasets with
# `completion_only_loss`. Rather than pin a version, detect what the installed TRL
# actually exposes so this adapts on the cluster instead of dying at job start.
# ------------------------------------------------------------------
_SFT_FIELDS = {f.name for f in dataclasses.fields(SFTConfig)}
COMPLETION_ONLY = "completion_only_loss" in _SFT_FIELDS
_LEN_FIELD = "max_seq_length" if "max_seq_length" in _SFT_FIELDS else "max_length"

if COMPLETION_ONLY:
    print(f"TRL supports completion_only_loss; length field is `{_LEN_FIELD}`.")
else:
    print(
        "\n" + "!" * 70 +
        "\n  WARNING: installed TRL has no `completion_only_loss`."
        "\n  Falling back to full-sequence loss on a `text` field. eval_loss is then"
        "\n  ~99.7% prompt reconstruction, so checkpoint selection and early stopping"
        "\n  on it are meaningless -- both are disabled and epochs are fixed instead."
        "\n" + "!" * 70 + "\n"
    )


def sft_config(**kwargs):
    """Build an SFTConfig, mapping the sequence-length field to whatever TRL calls it."""
    kwargs[_LEN_FIELD] = MAX_SEQ_LEN
    if COMPLETION_ONLY:
        kwargs["completion_only_loss"] = True
    return SFTConfig(**kwargs)


def make_dataset(df):
    """Prompt/completion pairs so the loss lands on the answer digits only.

    With a ~1030-token system prompt and a 3-token answer, full-sequence loss puts
    ~99.7% of the signal on reproducing the prompt.
    """
    if COMPLETION_ONLY:
        return Dataset.from_dict({
            "prompt": [format_prompt(r["review_text"]) for _, r in df.iterrows()],
            "completion": [
                format_answer(r["schedule_related"], r["job_control_related"])
                for _, r in df.iterrows()
            ],
        })
    return Dataset.from_dict({
        "text": [
            format_example(r["review_text"], r["schedule_related"], r["job_control_related"])
            for _, r in df.iterrows()
        ]
    })


def compute_fold_metrics(model, tokenizer, val_df, tag):
    """Greedy two-digit inference on val_df; return per-dimension + macro F1.

    Logs the first few raw generations and dumps all of them to disk. The previous run
    reported 100% malformed for 20 folds straight with no way to see what the model had
    actually emitted.
    """
    model.eval()
    ps, pc, raws, malformed = [], [], [], 0
    for i, (_, row) in enumerate(val_df.iterrows()):
        text = generate_two_digit(model, tokenizer, row["review_text"], max_new_tokens=8)
        s, c, ok = parse_two_digit(text)
        if i < LOG_FIRST_N_GENERATIONS:
            print(f"    [gen {tag} {i}] "
                  f"gold={int(row['schedule_related'])}{int(row['job_control_related'])}  "
                  f"raw={text!r}  parsed=({s},{c})  ok={ok}  "
                  f"review={row['review_text'][:50]!r}")
        malformed += (not ok)
        ps.append(s)
        pc.append(c)
        raws.append(text)

    frac = malformed / max(1, len(val_df))

    # Dumped outside the per-fold dirs, which the objective rmtree's for all but the best fold.
    gen_dir = os.path.join(RESULTS_DIR, "generations")
    os.makedirs(gen_dir, exist_ok=True)
    pd.DataFrame({
        "id": val_df["id"].tolist(),
        "review_text": val_df["review_text"].tolist(),
        "gold_sched": val_df["schedule_related"].tolist(),
        "gold_ctrl": val_df["job_control_related"].tolist(),
        "raw_output": raws,
        "sched_pred": ps,
        "ctrl_pred": pc,
    }).to_csv(os.path.join(gen_dir, f"{tag}.csv"), index=False)

    if frac > MALFORMED_ABORT_FRAC:
        raise DegenerateOutputError(
            f"{malformed}/{len(val_df)} ({frac:.0%}) generations unparseable in {tag}.\n"
            f"  First raw outputs: {raws[:5]!r}\n"
            f"  This is a setup bug (truncation / prompt format), not a hyperparameter."
        )

    ts = val_df["schedule_related"].tolist()
    tc = val_df["job_control_related"].tolist()
    sched_f1 = f1_score(ts, ps, pos_label=1, zero_division=0)
    ctrl_f1 = f1_score(tc, pc, pos_label=1, zero_division=0)
    return {
        "schedule_f1": float(sched_f1),
        "jobcontrol_f1": float(ctrl_f1),
        "macro_f1": float(0.5 * (sched_f1 + ctrl_f1)),
        "malformed": int(malformed),
        "malformed_frac": float(frac),
    }


def _load_lora(config):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    # Honour UNSLOTH_COMPILE_LOCATION so concurrent array tasks clear their own cache,
    # not each other's (a shared cache raced and killed tasks mid-compile).
    cache_dir = os.environ.get("UNSLOTH_COMPILE_LOCATION") or os.path.join(os.getcwd(), "unsloth_compiled_cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        load_in_16bit=False,
        full_finetuning=False,
        # "auto" measures free VRAM and, once fragmentation from many sequential
        # load/unload cycles eats into it, silently offloads some modules to CPU/disk
        # -- which 4-bit bitsandbytes weights can't do, raising ValueError (not
        # OutOfMemoryError) and killing the whole Optuna study. "balanced" always
        # splits across both GPUs instead of falling back to CPU/disk. Matches
        # 12.evaluate_multilabel.py, which already uses "balanced" for this reason.
        device_map="balanced",
    )
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer

    model = FastModel.get_peft_model(
        model,
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=0.0,  # MoE ParamWrapper (expert weights) require dropout=0
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=123,
        max_seq_length=MAX_SEQ_LEN,
        target_modules=TARGET_MODULES,
    )
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model, tokenizer


def assert_trainer_dataset(trainer):
    """Inspect what the trainer actually produced, not what we asked for.

    Immune to SFTConfig field renames and to whatever Unsloth's patched SFTTrainer does
    with prompt/completion datasets -- this reads the tokenized result.
    """
    ds = trainer.train_dataset
    cols = getattr(ds, "column_names", []) or []
    if "input_ids" not in cols:
        print(f"  [check] SKIPPED: train_dataset has no input_ids (columns={cols})")
        return

    lengths = [len(x) for x in ds["input_ids"]]
    n_at_cap = sum(n >= MAX_SEQ_LEN for n in lengths)
    print(f"  [check] tokenized train len max={max(lengths)} cap={MAX_SEQ_LEN} at_cap={n_at_cap}")
    if n_at_cap:
        raise SystemExit(
            f"FATAL: TRL truncated {n_at_cap} training example(s) at {MAX_SEQ_LEN} tokens.\n"
            f"  Raise MAX_SEQ_LEN in code/multilabel_prompt.py."
        )

    if "labels" in cols:
        labels = ds[0]["labels"]
        n_sup = sum(1 for t in labels if t != -100)
        print(f"  [check] supervised tokens in example 0: {n_sup} / {len(labels)}")
        if COMPLETION_ONLY and n_sup > ANSWER_MARGIN:
            raise SystemExit(
                f"FATAL: completion_only_loss=True but {n_sup} tokens are supervised "
                f"(expected ~3: two digits + <|im_end|>).\n"
                f"  The masking did not take effect -- loss would be dominated by the prompt."
            )


def _teardown(model, tokenizer, trainer):
    trainer.model = None
    trainer.optimizer = None
    trainer.lr_scheduler = None
    for cb in trainer.callback_handler.callbacks:
        if hasattr(cb, "model"):
            cb.model = None
    del model, tokenizer, trainer
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); gc.collect()


def train_fold(train_df, val_df, config, fold_idx, output_dir, tag):
    """Train one CV fold; return the metrics dict on the best-loss checkpoint."""
    print(f"\n{'='*60}")
    print(f"  Outer {OUTER_FOLD} / inner fold={fold_idx+1}/{N_FOLDS}  |  rank={config['lora_rank']}, "
          f"alpha={config['lora_alpha']}, lr={config['learning_rate']:.2e}")
    print(f"  Train: {len(train_df)} (oversampled), Val: {len(val_df)}")
    print(f"{'='*60}\n")

    model, tokenizer = _load_lora(config)
    # eval_loss only means something when the loss is masked to the answer tokens.
    select_on_eval_loss = COMPLETION_ONLY
    trainer_kwargs = {}
    if COMPLETION_ONLY:
        trainer_kwargs["callbacks"] = [EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
    else:
        trainer_kwargs["dataset_text_field"] = "text"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=make_dataset(train_df),
        eval_dataset=make_dataset(val_df),
        args=sft_config(
            per_device_train_batch_size=TRAIN_BS,
            per_device_eval_batch_size=EVAL_BS,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_steps=10,
            num_train_epochs=N_EPOCHS,
            learning_rate=config["learning_rate"],
            optim="adamw_8bit",
            bf16=True,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps" if select_on_eval_loss else "no",
            # scavenger partitions preempt without warning, so checkpoint every eval
            # rather than every other one. LoRA checkpoints are a few MB.
            save_steps=50,
            save_total_limit=2,
            output_dir=output_dir,
            seed=3407,
            dataloader_num_workers=0,
            dataset_num_proc=1,
            load_best_model_at_end=select_on_eval_loss,
            metric_for_best_model="eval_loss" if select_on_eval_loss else None,
            prediction_loss_only=True,
        ),
        **trainer_kwargs,
    )
    assert_trainer_dataset(trainer)
    # Resume this fold from its last checkpoint if a previous job was killed mid-fold.
    # Without this, a 24h timeout or a scavenger preemption threw away the whole fold.
    _ckpt = None
    if os.path.isdir(output_dir):
        _cks = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if _cks:
            _ckpt = True
            print(f"  [resume] found {len(_cks)} checkpoint(s) in {output_dir}; resuming")
    trainer.train(resume_from_checkpoint=_ckpt)

    metrics = compute_fold_metrics(model, tokenizer, val_df, tag)
    print(f"\n  Fold metrics: schedule_f1={metrics['schedule_f1']:.4f}  "
          f"jobcontrol_f1={metrics['jobcontrol_f1']:.4f}  macro_f1={metrics['macro_f1']:.4f}  "
          f"malformed={metrics['malformed']} ({metrics['malformed_frac']:.1%})  "
          f"|  stopped at epoch {trainer.state.epoch:.2f}")

    fold_adapter_dir = os.path.join(output_dir, "best_adapter")
    os.makedirs(fold_adapter_dir, exist_ok=True)
    model.save_pretrained(fold_adapter_dir)
    tokenizer.save_pretrained(fold_adapter_dir)

    _teardown(model, tokenizer, trainer)
    return metrics


def train_final(train_df, config):
    """Retrain on the full train split with the best config; save to FINAL_DIR."""
    train_df = maybe_oversample(train_df)
    print(f"\n{'='*60}")
    print(f"  FINAL TRAINING on full train split ({len(train_df)} rows, oversampled)")
    print(f"  r={config['lora_rank']}, alpha={config['lora_alpha']}, lr={config['learning_rate']:.2e}")
    print(f"{'='*60}\n")

    model, tokenizer = _load_lora(config)
    os.makedirs(FINAL_DIR, exist_ok=True)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=make_dataset(train_df),
        args=sft_config(
            per_device_train_batch_size=TRAIN_BS,
            per_device_eval_batch_size=EVAL_BS,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_steps=10,
            num_train_epochs=N_EPOCHS,
            learning_rate=config["learning_rate"],
            optim="adamw_8bit",
            bf16=True,
            logging_steps=5,
            eval_strategy="no",
            save_strategy="no",
            output_dir=FINAL_DIR,
            seed=3407,
            dataloader_num_workers=0,
            dataset_num_proc=1,
        ),
        **({} if COMPLETION_ONLY else {"dataset_text_field": "text"}),
    )
    assert_trainer_dataset(trainer)
    trainer.train()
    model.save_pretrained(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    print(f"\n  Final model saved to {FINAL_DIR}")
    _teardown(model, tokenizer, trainer)


# ============================================================
# Main: Optuna hyperparameter search with stratified K-fold CV
# ============================================================
print("Loading data...")
df = pd.read_csv(DATA_CSV)
if "fold" not in df.columns:
    raise SystemExit(
        f"{DATA_CSV} has no `fold` column. Rebuild it with 10.prepare_multilabel_data.py "
        f"(the old `set` split is a different, frozen artifact at labelled.csv)."
    )
n_outer = int(df.loc[df["fold"] >= 0, "fold"].nunique())
if OUTER_FOLD not in set(df["fold"]):
    raise SystemExit(f"ML_OUTER_FOLD={OUTER_FOLD} is not a fold in {DATA_CSV} "
                     f"(folds present: {sorted(set(df['fold']))})")

# fold == -1 are the pinned few-shot rows: always train, never test.
train_pool = df[df["fold"] != OUTER_FOLD].reset_index(drop=True)
n_test = int((df["fold"] == OUTER_FOLD).sum())
print(f"OUTER FOLD {OUTER_FOLD} of {n_outer}  |  total: {len(df)}  |  "
      f"train pool: {len(train_pool)}  |  held-out outer test: {n_test}  "
      f"(pinned few-shot in train: {int((df['fold'] == -1).sum())})")
print("Train-pool state counts:", train_pool["strat"].value_counts().sort_index().to_dict())

OVERSAMPLE_FACTORS = compute_oversample_factors(train_pool) if OVERSAMPLE else {}

# Whole df, not just train_pool, so 12.evaluate_multilabel.py's test rows are covered too.
preflight_token_budget(df)

os.makedirs(RESULTS_DIR, exist_ok=True)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=123)


def objective(trial):
    # Attention-only LoRA is ~98% smaller than the expert-inclusive variant, so these
    # ranks are all affordable where the old space OOM'd at r=16.
    rank = trial.suggest_categorical("r", [8, 16, 32, 64])
    lr = trial.suggest_float("lr", 5e-5, 3e-4, log=True)
    alpha_mode = trial.suggest_categorical("alpha_mode", ["r", "2r"])
    alpha = rank if alpha_mode == "r" else 2 * rank
    config = {"lora_rank": rank, "lora_alpha": alpha, "learning_rate": lr}

    print(f"\n{'*'*60}\n  Trial {trial.number}  |  r={rank}, alpha={alpha}, lr={lr:.2e}\n{'*'*60}")

    sched_f1s, macro_f1s, ctrl_f1s, malf_fracs = [], [], [], []
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(train_pool, train_pool["strat"])):
        tr_df = maybe_oversample(train_pool.iloc[tr_idx])
        va_df = train_pool.iloc[va_idx]  # never oversampled
        fold_dir = os.path.join(RESULTS_DIR, f"trial_{trial.number}", f"fold_{fold_idx}")
        tag = f"trial{trial.number}_fold{fold_idx}"
        m = train_fold(tr_df, va_df, config, fold_idx, fold_dir, tag)
        sched_f1s.append(m["schedule_f1"])
        ctrl_f1s.append(m["jobcontrol_f1"])
        macro_f1s.append(m["macro_f1"])
        malf_fracs.append(m["malformed_frac"])

    # keep only the best (by schedule F1) fold's adapter
    best_fold = int(np.argmax(sched_f1s))
    for fold_idx in range(N_FOLDS):
        if fold_idx != best_fold:
            d = os.path.join(RESULTS_DIR, f"trial_{trial.number}", f"fold_{fold_idx}")
            if os.path.exists(d):
                shutil.rmtree(d)

    trial.set_user_attr("fold_schedule_f1", sched_f1s)
    trial.set_user_attr("fold_jobcontrol_f1", ctrl_f1s)
    trial.set_user_attr("fold_malformed_frac", malf_fracs)
    trial.set_user_attr("mean_macro_f1", float(np.mean(macro_f1s)))
    trial.set_user_attr("mean_jobcontrol_f1", float(np.mean(ctrl_f1s)))
    mean_sched = float(np.mean(sched_f1s))
    print(f"\n  Trial {trial.number} done  |  mean_schedule_f1={mean_sched:.4f}  "
          f"mean_macro_f1={np.mean(macro_f1s):.4f}  mean_jobctrl_f1={np.mean(ctrl_f1s):.4f}  "
          f"max_malformed_frac={max(malf_fracs):.1%}")
    return float(np.mean(macro_f1s))  # primary objective: macro F1 (job-control is the binding dimension)


# Resumable study so scavenger-gpu preemption doesn't discard search progress. The outer
# fold is part of the key: five array tasks share a filesystem and must not share a study.
_study_key = f"{RECIPE_VERSION}_outer{OUTER_FOLD}" + (f"_{STUDY_TAG}" if STUDY_TAG else "")
STUDY_NAME = f"multilabel_lora_{_study_key}"
storage = f"sqlite:///{os.path.join(RESULTS_DIR, f'optuna_study_{_study_key}.db')}"
study = optuna.create_study(
    direction="maximize", study_name=STUDY_NAME,
    storage=storage, load_if_exists=True,
)

# Resuming is only safe if the recipe is unchanged. Without this, editing the prompt or
# the sequence length silently continues a search run under different conditions.
# data_sha1 covers the split itself, not just the recipe: re-running script 10 with a
# different seed, or with more coded rows, produces different folds, and resuming a study
# whose trials were scored against the old ones would be silently meaningless.
DATA_SHA1 = hashlib.sha1(
    df[["id", "fold", "strat"]].sort_values("id").to_csv(index=False).encode("utf-8")
).hexdigest()[:12]

FINGERPRINT = json.dumps({
    "outer_fold": OUTER_FOLD,
    "data_sha1": DATA_SHA1,
    "max_seq_len": MAX_SEQ_LEN,
    "target_modules": TARGET_MODULES,
    "completion_only": COMPLETION_ONLY,
    "prompt_sha1": hashlib.sha1(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12],
    "oversample_factors": {str(k): v for k, v in OVERSAMPLE_FACTORS.items()},
    "n_folds": N_FOLDS,
    "n_epochs": N_EPOCHS,
}, sort_keys=True)
_prev = study.user_attrs.get("fingerprint")
if _prev is None:
    study.set_user_attr("fingerprint", FINGERPRINT)
elif _prev != FINGERPRINT:
    raise SystemExit(
        f"Refusing to resume study {STUDY_NAME!r}: the training recipe changed.\n"
        f"  stored : {_prev}\n  current: {FINGERPRINT}\n"
        f"Bump RECIPE_VERSION to start a fresh study."
    )

n_complete = len([t for t in study.trials if t.state == TrialState.COMPLETE])
remaining = max(0, N_TRIALS - n_complete)
print(f"Optuna: {n_complete} complete trial(s) found, running {remaining} more.")
if remaining:
    # OOM only. torch.cuda.OutOfMemoryError is already a RuntimeError subclass, so also
    # listing RuntimeError/ValueError/NotImplementedError just hid every real bug in the
    # script as a FAIL trial. DegenerateOutputError subclasses Exception directly and so
    # cannot be caught here either -- a systemic failure should kill the study.
    study.optimize(objective, n_trials=remaining, catch=(torch.cuda.OutOfMemoryError,))

# ============================================================
# Select best config (schedule F1 primary, macro F1 tie-break) and retrain
# ============================================================
print("\n" + "=" * 60 + "\nOPTUNA SEARCH RESULTS\n" + "=" * 60)
# State filter, not just `value is not None`: a PRUNED trial can carry a value.
completed = [t for t in study.trials
             if t.state == TrialState.COMPLETE and t.value is not None]
for t in sorted(completed, key=lambda x: x.value, reverse=True):
    print(f"  trial={t.number}  mean_schedule_f1={t.value:.4f}  "
          f"mean_macro_f1={t.user_attrs.get('mean_macro_f1', 0):.4f}  params={t.params}")

if not completed:
    raise SystemExit("No trial completed. Refusing to train a final model.")

best_trial = max(completed, key=lambda t: (round(t.value, 4), t.user_attrs.get("mean_jobcontrol_f1", 0.0)))

n_tied = sum(1 for t in completed if round(t.value, 4) == round(best_trial.value, 4))
if n_tied > 1:
    print(f"  WARNING: {n_tied} trials tie at {best_trial.value:.4f}; "
          f"selection fell back to the macro-F1 tie-break.")

best_rank = best_trial.params["r"]
best_config = {
    "lora_rank": best_rank,
    "lora_alpha": best_rank if best_trial.params["alpha_mode"] == "r" else 2 * best_rank,
    "learning_rate": best_trial.params["lr"],
}
print(f"\nBest: trial={best_trial.number}  schedule_f1={best_trial.value:.4f}  "
      f"macro_f1={best_trial.user_attrs.get('mean_macro_f1', 0):.4f}  config={best_config}")

degenerate = best_trial.value < MIN_USABLE_F1

results_log = {
    "timestamp": datetime.now().isoformat(),
    "model": MODEL_NAME,
    "recipe_version": RECIPE_VERSION,
    "outer_fold": OUTER_FOLD,
    "data_sha1": DATA_SHA1,
    "n_outer_train": int(len(train_pool)),
    "n_outer_test": n_test,
    "max_seq_len": MAX_SEQ_LEN,
    "target_modules": TARGET_MODULES,
    "completion_only_loss": COMPLETION_ONLY,
    "n_folds": N_FOLDS,
    "n_trials": N_TRIALS,
    "n_epochs": N_EPOCHS,
    "oversample": OVERSAMPLE,
    "oversample_factors": OVERSAMPLE_FACTORS,
    "objective": "mean macro F1 (job-control-F1 tie-break)",
    "best_trial": best_trial.number,
    "best_params": best_trial.params,
    "best_mean_schedule_f1": best_trial.value,
    "best_mean_macro_f1": best_trial.user_attrs.get("mean_macro_f1"),
    "degenerate": bool(degenerate),
    "all_trials": [
        {
            "trial": t.number,
            "params": t.params,
            "mean_schedule_f1": t.value,
            "fold_schedule_f1": t.user_attrs.get("fold_schedule_f1", []),
            "fold_jobcontrol_f1": t.user_attrs.get("fold_jobcontrol_f1", []),
            "fold_malformed_frac": t.user_attrs.get("fold_malformed_frac", []),
            "mean_macro_f1": t.user_attrs.get("mean_macro_f1"),
        }
        for t in study.trials
    ],
}
_results_name = f"cv_results_{_study_key}.json"
with open(os.path.join(RESULTS_DIR, _results_name), "w") as f:
    json.dump(results_log, f, indent=2, default=str)

print(f"\nResults saved to {RESULTS_DIR}/{_results_name}")
print(f"Raw generations in {RESULTS_DIR}/generations/")

# ============================================================
# Final training on the full train split
# ============================================================
# The previous run picked a winner from a three-way tie at 0.0000 and spent a day
# training a final model on an arbitrary config. Every config failing identically is a
# setup bug, not a search result.
_degenerate_msg = (
    f"Best mean schedule F1 = {best_trial.value:.4f} < {MIN_USABLE_F1}.\n"
    f"  Every config failed similarly -> this is a setup bug, not a search result.\n"
    f"  Malformed fractions per trial: "
    f"{[t.user_attrs.get('fold_malformed_frac') for t in completed]}\n"
    f"  Inspect {RESULTS_DIR}/generations/*.csv for the raw model output."
)

if SKIP_FINAL:
    if degenerate:
        print(f"\nWARNING: {_degenerate_msg}")
    print("\nML_SKIP_FINAL set: skipping final training, leaving FINAL_DIR untouched.")
elif degenerate:
    raise SystemExit(f"{_degenerate_msg}\n  Refusing to spend GPU-days training a final model.")
else:
    if os.path.exists(FINAL_DIR):
        shutil.rmtree(FINAL_DIR)
    train_final(train_pool, best_config)
    print(f"Final model saved to {FINAL_DIR}")

open(os.path.join(RESULTS_DIR, "DONE"), "w").write(datetime.now().isoformat())
print("Done!")
