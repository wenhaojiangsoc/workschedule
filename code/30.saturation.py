"""
Learning curve: held-out F1 as a function of the number of hand-coded training rows.

Answers the question the hyperparameter search cannot: is the model limited by the
recipe or by the amount of labelled data? If F1 is still climbing at the full training
size, more coding pays. If it has flattened, it does not.

One task per model, chosen with ML_SAT_TASK:
    s1  the schedule/job-control gate      (2 digits, metric = macro F1)
    sc  20 schedule dimensions             (metric = micro F1 over dimension decisions)
    jc  7 job-control dimensions           (metric = micro F1 over dimension decisions)

Outer fold 0 is the fixed held-out test set for every point on the curve, so the only
thing that changes along the x axis is how much training data the model saw. Training
subsets are drawn stratified and are nested across sizes within a seed, so the curve is
not confounded by which rows happened to be picked.

Hyperparameters are FIXED (no Optuna): a learning curve must vary one thing at a time.

Run:  ML_SAT_TASK=s1 python code/30.saturation.py
"""
import os

TASK = os.environ.get("ML_SAT_TASK", "s1")
if TASK not in ("s1", "sc", "jc"):
    raise SystemExit("ML_SAT_TASK must be s1, sc or jc")

import sys
import json
import gc
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from unsloth import FastModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
import dataclasses

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TEST_FOLD = 0
SEEDS = [int(x) for x in os.environ.get("ML_SAT_SEEDS", "0,1").split(",")]
LORA_RANK = int(os.environ.get("ML_SAT_RANK", 16))
LORA_ALPHA = LORA_RANK
LEARNING_RATE = float(os.environ.get("ML_SAT_LR", 1e-4))
N_EPOCHS = float(os.environ.get("ML_SAT_EPOCHS", 5))
TRAIN_BS = int(os.environ.get("ML_TRAIN_BS", 8))
GRAD_ACCUM = int(os.environ.get("ML_GRAD_ACCUM", 1))
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

if TASK == "s1":
    from multilabel_prompt import (MODEL_NAME, MODEL_SLUG, MAX_SEQ_LEN, format_prompt,
                                   format_answer, format_example, generate_two_digit)
    from metrics_multilabel import parse_two_digit
    DATA_CSV = "data/trainingfinal/labelled_cv.csv"
    LABELS = ["schedule_related", "job_control_related"]
else:
    from hier_prompt import (MODEL_NAME, MODEL_SLUG, MAX_SEQ_LEN, dims_for, n_digits,
                             format_prompt as h_prompt, format_answer as h_answer,
                             format_example as h_example, parse_digits, generate_digits)
    DATA_CSV = f"data/trainingfinal/hier_{TASK}_cv.csv"
    LABELS = dims_for(TASK)

OUT_DIR = f"models/saturation/{MODEL_SLUG}/{TASK}"
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_JSON = os.path.join(OUT_DIR, f"saturation_{TASK}.json")

_SFT = {f.name for f in dataclasses.fields(SFTConfig)}
COMPLETION_ONLY = "completion_only_loss" in _SFT
_LEN_FIELD = "max_seq_length" if "max_seq_length" in _SFT else "max_length"


def sft_config(**kw):
    kw[_LEN_FIELD] = MAX_SEQ_LEN
    if COMPLETION_ONLY:
        kw["completion_only_loss"] = True
    return SFTConfig(**kw)


def make_dataset(df):
    if TASK == "s1":
        if COMPLETION_ONLY:
            return Dataset.from_dict({
                "prompt": [format_prompt(r["review_text"]) for _, r in df.iterrows()],
                "completion": [format_answer(r["schedule_related"], r["job_control_related"])
                               for _, r in df.iterrows()]})
        return Dataset.from_dict({"text": [format_example(r["review_text"], r["schedule_related"],
                                                          r["job_control_related"])
                                           for _, r in df.iterrows()]})
    if COMPLETION_ONLY:
        return Dataset.from_dict({
            "prompt": [h_prompt(TASK, r["review_text"], r["type"]) for _, r in df.iterrows()],
            "completion": [h_answer(str(r["target"])) for _, r in df.iterrows()]})
    return Dataset.from_dict({"text": [h_example(TASK, r["review_text"], r["type"], str(r["target"]))
                                       for _, r in df.iterrows()]})


def evaluate(model, tokenizer, test_df):
    if TASK == "s1":
        ps, pc, malformed = [], [], 0
        for _, row in test_df.iterrows():
            t = generate_two_digit(model, tokenizer, row["review_text"], max_new_tokens=8)
            s, c, ok = parse_two_digit(t)
            malformed += (not ok); ps.append(s); pc.append(c)
        sf = f1_score(test_df["schedule_related"], ps, pos_label=1, zero_division=0)
        cf = f1_score(test_df["job_control_related"], pc, pos_label=1, zero_division=0)
        return {"schedule_f1": float(sf), "jobcontrol_f1": float(cf),
                "macro_f1": float(0.5 * (sf + cf)), "primary": float(0.5 * (sf + cf)),
                "malformed": malformed}
    nd = n_digits(TASK)
    preds, malformed = [], 0
    for _, row in test_df.iterrows():
        t = generate_digits(model, tokenizer, TASK, row["review_text"], row["type"])
        d, ok = parse_digits(t, TASK)
        malformed += (not ok); preds.append(d)
    y = np.array([[int(c) for c in str(t).zfill(nd)] for t in test_df["target"]])
    p = np.array(preds)
    per = {d: float(f1_score(y[:, j], p[:, j], pos_label=1, zero_division=0))
           for j, d in enumerate(LABELS)}
    micro = float(f1_score(y.ravel(), p.ravel(), pos_label=1, zero_division=0))
    sup = y.sum(axis=0)
    macro = float(np.mean([per[d] for j, d in enumerate(LABELS) if sup[j] > 0]))
    return {"micro_f1": micro, "macro_f1": macro, "primary": micro,
            "per_dim_f1": per, "support": {d: int(sup[j]) for j, d in enumerate(LABELS)},
            "malformed": malformed}


def train_once(train_df, test_df, tag):
    gc.collect(); torch.cuda.empty_cache()
    cache = os.environ.get("UNSLOTH_COMPILE_LOCATION") or os.path.join(os.getcwd(), "unsloth_compiled_cache")
    if os.path.exists(cache):
        shutil.rmtree(cache)
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True,
        load_in_16bit=False, full_finetuning=False, device_map="balanced")
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    model = FastModel.get_peft_model(
        model, r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=123,
        max_seq_length=MAX_SEQ_LEN, target_modules=TARGET_MODULES)
    out = os.path.join(OUT_DIR, "work", tag)
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=make_dataset(train_df),
        args=sft_config(per_device_train_batch_size=TRAIN_BS,
                        gradient_accumulation_steps=GRAD_ACCUM, warmup_steps=5,
                        num_train_epochs=N_EPOCHS, learning_rate=LEARNING_RATE,
                        optim="adamw_8bit", bf16=True, logging_steps=20,
                        eval_strategy="no",
                        # No eval here (the curve is scored once at the end), but the run
                        # still needs checkpoints: without them a preemption discarded the
                        # whole point. save_steps does not have to align with eval when
                        # load_best_model_at_end is off.
                        save_strategy="steps", save_steps=50, save_total_limit=1,
                        output_dir=out, seed=3407,
                        dataloader_num_workers=0, dataset_num_proc=1),
        **({} if COMPLETION_ONLY else {"dataset_text_field": "text"}))
    _ckpt = None
    if os.path.isdir(out) and any(d.startswith("checkpoint-") for d in os.listdir(out)):
        _ckpt = True
        print(f"  [resume] checkpoint found in {out}; resuming this curve point")
    trainer.train(resume_from_checkpoint=_ckpt)
    m = evaluate(model, tokenizer, test_df)
    trainer.model = None; trainer.optimizer = None; trainer.lr_scheduler = None
    del model, tokenizer, trainer
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    if os.path.exists(out):
        shutil.rmtree(out, ignore_errors=True)
    return m


def stratum(df):
    if TASK == "s1":
        return (2 * df["schedule_related"] + df["job_control_related"]).astype(str)
    return df[LABELS].sum(axis=1).clip(upper=3).astype(str) + df["type"].astype(str)


def nested_subsets(pool, sizes, seed):
    """Nested stratified subsets: the n=200 set contains the n=100 set."""
    rng = np.random.RandomState(seed)
    strat = stratum(pool)
    order = []
    for s in strat.unique():
        # .to_numpy() on an Index can return a read-only view, which rng.shuffle
        # refuses to touch ("ValueError: array is read-only"). Copy before shuffling.
        idx = np.array(pool.index[strat == s], dtype=np.int64)
        rng.shuffle(idx)
        order.append(idx)
    # round-robin across strata so every prefix stays roughly balanced
    seq, k = [], 0
    while len(seq) < len(pool):
        for arr in order:
            if k < len(arr):
                seq.append(arr[k])
        k += 1
    seq = np.array(seq[:len(pool)])
    return {n: pool.loc[seq[:n]] for n in sizes}


df = pd.read_csv(DATA_CSV, dtype={"target": str}) if TASK != "s1" else pd.read_csv(DATA_CSV)
if TASK != "s1":
    df["target"] = df["target"].str.zfill(n_digits(TASK))
test_df = df[df["fold"] == TEST_FOLD].reset_index(drop=True)
pool = df[df["fold"] != TEST_FOLD].reset_index(drop=True)
N = len(pool)
raw_sizes = [100, 200, 300, 450, 600, 800, 1000, 1266]
SIZES = sorted({s for s in raw_sizes if s < N} | {N})
print(f"TASK={TASK}  pool={N}  test(fold {TEST_FOLD})={len(test_df)}  sizes={SIZES}  seeds={SEEDS}")

results = []
if os.path.exists(RESULTS_JSON):
    results = json.load(open(RESULTS_JSON)).get("points", [])
    print(f"resuming: {len(results)} point(s) already done")
done = {(r["n"], r["seed"]) for r in results}

for seed in SEEDS:
    subsets = nested_subsets(pool, SIZES, seed)
    for n in SIZES:
        if (n, seed) in done:
            continue
        tag = f"{TASK}_n{n}_s{seed}"
        print(f"\n{'='*60}\n  {tag}: training on {n} rows, testing on {len(test_df)}\n{'='*60}")
        m = train_once(subsets[n], test_df, tag)
        m.update({"n": n, "seed": seed, "task": TASK})
        results.append(m)
        print(f"  -> primary={m['primary']:.4f}  malformed={m['malformed']}")
        json.dump({"task": TASK, "model": MODEL_NAME, "test_fold": TEST_FOLD,
                   "n_test": len(test_df), "sizes": SIZES, "seeds": SEEDS,
                   "lora_rank": LORA_RANK, "lr": LEARNING_RATE, "epochs": N_EPOCHS,
                   "timestamp": datetime.now().isoformat(), "points": results},
                  open(RESULTS_JSON, "w"), indent=2)

print("\n=== learning curve ===")
t = pd.DataFrame([{k: v for k, v in r.items() if k in ("n", "seed", "primary")} for r in results])
if len(t):
    g = t.groupby("n")["primary"].agg(["mean", "std", "count"])
    for n, row in g.iterrows():
        print(f"  n={n:>5}  F1={row['mean']:.4f}  sd={0 if np.isnan(row['std']) else row['std']:.4f}  (k={int(row['count'])})")
open(os.path.join(OUT_DIR, "DONE"), "w").write(datetime.now().isoformat())
print("Done!")
