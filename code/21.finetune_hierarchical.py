"""
Stage 2 of the hierarchical classifier: which dimensions, given the branch.

Same recipe as 11.finetune_multilabel.py (Qwen3.5-35B-A3B, 4-bit, attention-only LoRA,
Optuna over rank/alpha/lr, nested outer CV) but the target is an N-digit multi-label
string instead of two digits:
    ML_BRANCH=sc -> 20 schedule dimensions
    ML_BRANCH=jc ->  7 job-control dimensions

Folds are inherited from stage 1, so a review held out by the stage-1 gate is the same
review held out here. Rows at fold == -1 are the stage-1 few-shot rows: always train.

Objective is micro-F1 over all dimension decisions (stable when several dimensions are
rare), with macro-F1 as the tie-break. Per-dimension F1 is recorded for every trial so
the final report can say which dimensions are learnable and which need more labelling.

Run:  ML_BRANCH=sc ML_OUTER_FOLD=0 python code/21.finetune_hierarchical.py
"""

import os

for _v in ("ML_BRANCH", "ML_OUTER_FOLD"):
    if _v not in os.environ:
        raise SystemExit(f"{_v} is not set. Run e.g. ML_BRANCH=sc ML_OUTER_FOLD=0 python code/21.finetune_hierarchical.py")
BRANCH = os.environ["ML_BRANCH"]
OUTER_FOLD = int(os.environ["ML_OUTER_FOLD"])
if BRANCH not in ("sc", "jc"):
    raise SystemExit(f"ML_BRANCH must be 'sc' or 'jc', got {BRANCH!r}")

import sys
import json
import gc
import shutil
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import optuna
from optuna.trial import TrialState
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hier_prompt import (
    MODEL_NAME, MODEL_SLUG, MAX_SEQ_LEN, ANSWER_MARGIN, dims_for, n_digits,
    system_prompt, format_prompt, format_answer, format_example,
    parse_digits, generate_digits, preflight_token_budget,
)

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback
import dataclasses

DIMS = dims_for(BRANCH)
NDIG = n_digits(BRANCH)
DATA_CSV = f"data/trainingfinal/hier_{BRANCH}_cv.csv"
RESULTS_DIR = f"models/cv_results_hier/{MODEL_SLUG}/{BRANCH}/outer{OUTER_FOLD}"
FINAL_DIR = f"models/finetuned/{MODEL_SLUG}_hier_{BRANCH}/outer{OUTER_FOLD}"

RECIPE_VERSION = "h1"
N_FOLDS = int(os.environ.get("ML_N_FOLDS", 5))
N_TRIALS = int(os.environ.get("ML_N_TRIALS", 5))
N_EPOCHS = float(os.environ.get("ML_EPOCHS", 6))
PATIENCE = 3
TRAIN_BS = int(os.environ.get("ML_TRAIN_BS", 8))
GRAD_ACCUM = int(os.environ.get("ML_GRAD_ACCUM", 1))
EVAL_BS = int(os.environ.get("ML_EVAL_BS", 16))
STUDY_TAG = os.environ.get("ML_STUDY_TAG", "").strip()
SKIP_FINAL = os.environ.get("ML_SKIP_FINAL", "").strip() not in ("", "0")

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MALFORMED_ABORT_FRAC = 0.25
LOG_FIRST_N_GENERATIONS = 8
MIN_USABLE_F1 = 0.35          # micro-F1 floor; below this the setup, not the config, is wrong
OVERSAMPLE_CAP = 4
RARE_TARGET = 40              # replicate rows carrying a dimension with fewer positives


class DegenerateOutputError(Exception):
    """Generations unparseable at scale: a setup bug, not a hyperparameter."""


_SFT_FIELDS = {f.name for f in dataclasses.fields(SFTConfig)}
COMPLETION_ONLY = "completion_only_loss" in _SFT_FIELDS
_LEN_FIELD = "max_seq_length" if "max_seq_length" in _SFT_FIELDS else "max_length"
print(f"TRL completion_only_loss={COMPLETION_ONLY}; length field `{_LEN_FIELD}`.")


def sft_config(**kw):
    kw[_LEN_FIELD] = MAX_SEQ_LEN
    if COMPLETION_ONLY:
        kw["completion_only_loss"] = True
    return SFTConfig(**kw)


def oversample_factors(pool):
    """Replicate rows carrying rare dimensions, capped. Train split only."""
    counts = {d: int(pool[d].sum()) for d in DIMS}
    fac = {}
    for d, n in counts.items():
        if 0 < n < RARE_TARGET:
            fac[d] = int(np.clip(round(RARE_TARGET / max(n, 1)), 1, OVERSAMPLE_CAP))
    print(f"Oversampling rare dimensions (<{RARE_TARGET} positives): "
          f"{ {d: (counts[d], f) for d, f in fac.items() if f > 1} }")
    return {d: f for d, f in fac.items() if f > 1}


def maybe_oversample(df, factors):
    if not factors:
        return df
    parts = [df]
    for d, f in factors.items():
        sub = df[df[d] == 1]
        if len(sub):
            parts.extend([sub] * (f - 1))
    return pd.concat(parts, ignore_index=True)


def make_dataset(df):
    if COMPLETION_ONLY:
        return Dataset.from_dict({
            "prompt": [format_prompt(BRANCH, r["review_text"], r["type"]) for _, r in df.iterrows()],
            "completion": [format_answer(str(r["target"])) for _, r in df.iterrows()],
        })
    return Dataset.from_dict({
        "text": [format_example(BRANCH, r["review_text"], r["type"], str(r["target"]))
                 for _, r in df.iterrows()]
    })


def dimension_metrics(y_true, y_pred):
    """y_true/y_pred: (n, NDIG) arrays. Returns micro, macro and per-dimension F1."""
    per = {}
    for j, d in enumerate(DIMS):
        per[d] = float(f1_score(y_true[:, j], y_pred[:, j], pos_label=1, zero_division=0))
    micro = float(f1_score(y_true.ravel(), y_pred.ravel(), pos_label=1, zero_division=0))
    support = y_true.sum(axis=0)
    learnable = [per[d] for j, d in enumerate(DIMS) if support[j] > 0]
    macro = float(np.mean(learnable)) if learnable else 0.0
    return micro, macro, per, support


def compute_fold_metrics(model, tokenizer, val_df, tag):
    model.eval()
    preds, raws, malformed = [], [], 0
    for i, (_, row) in enumerate(val_df.iterrows()):
        text = generate_digits(model, tokenizer, BRANCH, row["review_text"], row["type"])
        d, ok = parse_digits(text, BRANCH)
        if i < LOG_FIRST_N_GENERATIONS:
            print(f"    [gen {tag} {i}] gold={row['target']}  raw={text!r}  ok={ok}  "
                  f"review={row['review_text'][:45]!r}")
        malformed += (not ok)
        preds.append(d)
        raws.append(text)
    frac = malformed / max(1, len(val_df))

    gen_dir = os.path.join(RESULTS_DIR, "generations")
    os.makedirs(gen_dir, exist_ok=True)
    pd.DataFrame({"id": val_df["id"].tolist(), "review_text": val_df["review_text"].tolist(),
                  "type": val_df["type"].tolist(), "gold": val_df["target"].astype(str).tolist(),
                  "raw_output": raws,
                  "pred": ["".join(map(str, p)) for p in preds]}).to_csv(
        os.path.join(gen_dir, f"{tag}.csv"), index=False)

    if frac > MALFORMED_ABORT_FRAC:
        raise DegenerateOutputError(
            f"{malformed}/{len(val_df)} ({frac:.0%}) generations unparseable in {tag}. "
            f"First raw outputs: {raws[:5]!r}")

    y_true = np.array([[int(c) for c in str(t).zfill(NDIG)] for t in val_df["target"]])
    y_pred = np.array(preds)
    micro, macro, per, support = dimension_metrics(y_true, y_pred)
    return {"micro_f1": micro, "macro_f1": macro, "per_dim_f1": per,
            "support": {d: int(support[j]) for j, d in enumerate(DIMS)},
            "malformed": int(malformed), "malformed_frac": float(frac)}


def _load_lora(config):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    # Honour UNSLOTH_COMPILE_LOCATION so concurrent array tasks clear their own cache,
    # not each other's (a shared cache raced and killed tasks mid-compile).
    cache_dir = os.environ.get("UNSLOTH_COMPILE_LOCATION") or os.path.join(os.getcwd(), "unsloth_compiled_cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True,
        load_in_16bit=False, full_finetuning=False, device_map="balanced")
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    model = FastModel.get_peft_model(
        model, r=config["lora_rank"], lora_alpha=config["lora_alpha"], lora_dropout=0.0,
        bias="none", use_gradient_checkpointing="unsloth", random_state=123,
        max_seq_length=MAX_SEQ_LEN, target_modules=TARGET_MODULES)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model, tokenizer


def assert_trainer_dataset(trainer):
    ds = trainer.train_dataset
    cols = getattr(ds, "column_names", []) or []
    if "input_ids" not in cols:
        print(f"  [check] SKIPPED: no input_ids (columns={cols})")
        return
    lengths = [len(x) for x in ds["input_ids"]]
    n_at_cap = sum(n >= MAX_SEQ_LEN for n in lengths)
    print(f"  [check] tokenized train len max={max(lengths)} cap={MAX_SEQ_LEN} at_cap={n_at_cap}")
    if n_at_cap:
        raise SystemExit(f"FATAL: TRL truncated {n_at_cap} example(s) at {MAX_SEQ_LEN}.")
    if "labels" in cols:
        n_sup = sum(1 for t in ds[0]["labels"] if t != -100)
        print(f"  [check] supervised tokens in example 0: {n_sup} / {len(ds[0]['labels'])}")
        if COMPLETION_ONLY and n_sup > NDIG + ANSWER_MARGIN:
            raise SystemExit(f"FATAL: completion masking did not take effect ({n_sup} supervised).")


def _teardown(model, tokenizer, trainer):
    trainer.model = None; trainer.optimizer = None; trainer.lr_scheduler = None
    for cb in trainer.callback_handler.callbacks:
        if hasattr(cb, "model"):
            cb.model = None
    del model, tokenizer, trainer
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); gc.collect()


def train_fold(train_df, val_df, config, output_dir, tag):
    print(f"\n{'='*60}\n  {BRANCH} outer{OUTER_FOLD} {tag}  |  r={config['lora_rank']}, "
          f"alpha={config['lora_alpha']}, lr={config['learning_rate']:.2e}")
    print(f"  Train: {len(train_df)} (oversampled), Val: {len(val_df)}\n{'='*60}\n")
    model, tokenizer = _load_lora(config)
    select = COMPLETION_ONLY
    kw = {"callbacks": [EarlyStoppingCallback(early_stopping_patience=PATIENCE)]} if select \
        else {"dataset_text_field": "text"}
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=make_dataset(train_df), eval_dataset=make_dataset(val_df),
        args=sft_config(
            per_device_train_batch_size=TRAIN_BS, per_device_eval_batch_size=EVAL_BS,
            gradient_accumulation_steps=GRAD_ACCUM, warmup_steps=10,
            num_train_epochs=N_EPOCHS, learning_rate=config["learning_rate"],
            optim="adamw_8bit", bf16=True, logging_steps=5,
            eval_strategy="steps", eval_steps=25,
            save_strategy="steps" if select else "no", save_steps=25, save_total_limit=2,
            output_dir=output_dir, seed=3407, dataloader_num_workers=0, dataset_num_proc=1,
            load_best_model_at_end=select,
            metric_for_best_model="eval_loss" if select else None,
            prediction_loss_only=True),
        **kw)
    assert_trainer_dataset(trainer)
    _ckpt = None
    if os.path.isdir(output_dir):
        _cks = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if _cks:
            _ckpt = True
            print(f"  [resume] found {len(_cks)} checkpoint(s) in {output_dir}; resuming")
    trainer.train(resume_from_checkpoint=_ckpt)
    m = compute_fold_metrics(model, tokenizer, val_df, tag)
    print(f"\n  {tag}: micro_f1={m['micro_f1']:.4f}  macro_f1={m['macro_f1']:.4f}  "
          f"malformed={m['malformed']} ({m['malformed_frac']:.1%})  "
          f"stopped at epoch {trainer.state.epoch:.2f}")
    _teardown(model, tokenizer, trainer)
    return m


def train_final(train_df, config, factors):
    train_df = maybe_oversample(train_df, factors)
    print(f"\n{'='*60}\n  FINAL {BRANCH} outer{OUTER_FOLD}: {len(train_df)} rows\n{'='*60}\n")
    model, tokenizer = _load_lora(config)
    os.makedirs(FINAL_DIR, exist_ok=True)
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=make_dataset(train_df),
        args=sft_config(
            per_device_train_batch_size=TRAIN_BS, per_device_eval_batch_size=EVAL_BS,
            gradient_accumulation_steps=GRAD_ACCUM, warmup_steps=10,
            num_train_epochs=N_EPOCHS, learning_rate=config["learning_rate"],
            optim="adamw_8bit", bf16=True, logging_steps=5, eval_strategy="no",
            save_strategy="no", output_dir=FINAL_DIR, seed=3407,
            dataloader_num_workers=0, dataset_num_proc=1),
        **({} if COMPLETION_ONLY else {"dataset_text_field": "text"}))
    assert_trainer_dataset(trainer)
    trainer.train()
    model.save_pretrained(FINAL_DIR); tokenizer.save_pretrained(FINAL_DIR)
    print(f"  Final model saved to {FINAL_DIR}")
    _teardown(model, tokenizer, trainer)


# ============================================================
print(f"Loading {DATA_CSV} ...")
df = pd.read_csv(DATA_CSV, dtype={"target": str})
df["target"] = df["target"].str.zfill(NDIG)
if OUTER_FOLD not in set(df["fold"]):
    raise SystemExit(f"fold {OUTER_FOLD} absent; folds present: {sorted(set(df['fold']))}")

train_pool = df[df["fold"] != OUTER_FOLD].reset_index(drop=True)
test_df = df[df["fold"] == OUTER_FOLD]
print(f"BRANCH {BRANCH} | OUTER FOLD {OUTER_FOLD} | total {len(df)} | "
      f"train pool {len(train_pool)} | held-out {len(test_df)} | pinned {(df['fold']==-1).sum()}")
print("Train-pool positives per dimension:")
for d in DIMS:
    print(f"   {int(train_pool[d].sum()):>4}  {d}")

OVERSAMPLE_FACTORS = oversample_factors(train_pool)
preflight_token_budget(BRANCH, df)

os.makedirs(RESULTS_DIR, exist_ok=True)
# Stratify inner folds on the number of positive dimensions, bucketed: keeps multi-dimension
# rows spread across folds without needing an iterative multi-label stratifier.
strat_key = train_pool[DIMS].sum(axis=1).clip(upper=3).astype(str) + train_pool["type"].astype(str)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=123)


def objective(trial):
    rank = trial.suggest_categorical("r", [8, 16, 32, 64])
    lr = trial.suggest_float("lr", 5e-5, 3e-4, log=True)
    alpha_mode = trial.suggest_categorical("alpha_mode", ["r", "2r"])
    alpha = rank if alpha_mode == "r" else 2 * rank
    config = {"lora_rank": rank, "lora_alpha": alpha, "learning_rate": lr}
    print(f"\n{'*'*60}\n  Trial {trial.number} | r={rank}, alpha={alpha}, lr={lr:.2e}\n{'*'*60}")

    micros, macros, malfs, per_dims = [], [], [], []
    for fold_idx, (tr, va) in enumerate(skf.split(train_pool, strat_key)):
        tr_df = maybe_oversample(train_pool.iloc[tr], OVERSAMPLE_FACTORS)
        va_df = train_pool.iloc[va]
        tag = f"trial{trial.number}_fold{fold_idx}"
        m = train_fold(tr_df, va_df, config,
                       os.path.join(RESULTS_DIR, f"trial_{trial.number}", f"fold_{fold_idx}"), tag)
        micros.append(m["micro_f1"]); macros.append(m["macro_f1"])
        malfs.append(m["malformed_frac"]); per_dims.append(m["per_dim_f1"])

    for fold_idx in range(N_FOLDS):
        d = os.path.join(RESULTS_DIR, f"trial_{trial.number}", f"fold_{fold_idx}")
        if os.path.exists(d):
            shutil.rmtree(d)

    mean_per_dim = {d: float(np.mean([p[d] for p in per_dims])) for d in DIMS}
    trial.set_user_attr("fold_micro_f1", micros)
    trial.set_user_attr("fold_macro_f1", macros)
    trial.set_user_attr("mean_macro_f1", float(np.mean(macros)))
    trial.set_user_attr("mean_per_dim_f1", mean_per_dim)
    trial.set_user_attr("fold_malformed_frac", malfs)
    print(f"\n  Trial {trial.number}: mean_micro_f1={np.mean(micros):.4f}  "
          f"mean_macro_f1={np.mean(macros):.4f}  max_malformed={max(malfs):.1%}")
    return float(np.mean(micros))


_key = f"{RECIPE_VERSION}_{BRANCH}_outer{OUTER_FOLD}" + (f"_{STUDY_TAG}" if STUDY_TAG else "")
study = optuna.create_study(direction="maximize", study_name=f"hier_{_key}",
                            storage=f"sqlite:///{os.path.join(RESULTS_DIR, f'optuna_{_key}.db')}",
                            load_if_exists=True)

DATA_SHA1 = hashlib.sha1(
    df[["id", "fold", "target"]].sort_values("id").to_csv(index=False).encode()).hexdigest()[:12]
FINGERPRINT = json.dumps({"branch": BRANCH, "outer_fold": OUTER_FOLD, "data_sha1": DATA_SHA1,
                          "max_seq_len": MAX_SEQ_LEN, "target_modules": TARGET_MODULES,
                          "completion_only": COMPLETION_ONLY,
                          "prompt_sha1": hashlib.sha1(system_prompt(BRANCH).encode()).hexdigest()[:12],
                          "oversample": {k: v for k, v in OVERSAMPLE_FACTORS.items()},
                          "n_folds": N_FOLDS, "n_epochs": N_EPOCHS}, sort_keys=True)
_prev = study.user_attrs.get("fingerprint")
if _prev is None:
    study.set_user_attr("fingerprint", FINGERPRINT)
elif _prev != FINGERPRINT:
    raise SystemExit(f"Refusing to resume: recipe changed.\n stored: {_prev}\n current: {FINGERPRINT}")

n_complete = len([t for t in study.trials if t.state == TrialState.COMPLETE])
remaining = max(0, N_TRIALS - n_complete)
print(f"Optuna: {n_complete} complete, running {remaining} more.")
if remaining:
    study.optimize(objective, n_trials=remaining, catch=(torch.cuda.OutOfMemoryError,))

completed = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
if not completed:
    raise SystemExit("No trial completed. Refusing to train a final model.")
for t in sorted(completed, key=lambda x: x.value, reverse=True):
    print(f"  trial={t.number} micro={t.value:.4f} macro={t.user_attrs.get('mean_macro_f1',0):.4f} {t.params}")

best = max(completed, key=lambda t: (round(t.value, 4), t.user_attrs.get("mean_macro_f1", 0.0)))
best_rank = best.params["r"]
best_config = {"lora_rank": best_rank,
               "lora_alpha": best_rank if best.params["alpha_mode"] == "r" else 2 * best_rank,
               "learning_rate": best.params["lr"]}
print(f"\nBest: trial={best.number} micro_f1={best.value:.4f} config={best_config}")

results = {"timestamp": datetime.now().isoformat(), "model": MODEL_NAME, "branch": BRANCH,
           "recipe_version": RECIPE_VERSION, "outer_fold": OUTER_FOLD, "data_sha1": DATA_SHA1,
           "dimensions": DIMS, "n_outer_train": len(train_pool), "n_outer_test": len(test_df),
           "train_pool_support": {d: int(train_pool[d].sum()) for d in DIMS},
           "objective": "mean micro F1 over dimension decisions (macro-F1 tie-break)",
           "best_trial": best.number, "best_params": best.params,
           "best_mean_micro_f1": best.value,
           "best_mean_macro_f1": best.user_attrs.get("mean_macro_f1"),
           "best_mean_per_dim_f1": best.user_attrs.get("mean_per_dim_f1"),
           "all_trials": [{"trial": t.number, "params": t.params, "mean_micro_f1": t.value,
                           "mean_macro_f1": t.user_attrs.get("mean_macro_f1"),
                           "mean_per_dim_f1": t.user_attrs.get("mean_per_dim_f1"),
                           "fold_malformed_frac": t.user_attrs.get("fold_malformed_frac")}
                          for t in study.trials]}
with open(os.path.join(RESULTS_DIR, f"cv_results_{_key}.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {RESULTS_DIR}/cv_results_{_key}.json")

if SKIP_FINAL:
    print("ML_SKIP_FINAL set: not training the final model.")
elif best.value < MIN_USABLE_F1:
    raise SystemExit(f"Best micro F1 {best.value:.4f} < {MIN_USABLE_F1}: setup bug, not a config. "
                     f"Inspect {RESULTS_DIR}/generations/*.csv")
else:
    if os.path.exists(FINAL_DIR):
        shutil.rmtree(FINAL_DIR)
    train_final(train_pool, best_config, OVERSAMPLE_FACTORS)
open(os.path.join(RESULTS_DIR, "DONE"), "w").write(datetime.now().isoformat())
print("Done!")
