"""
Score a mined candidate pool with the trained two-stage classifier.

Keyword mining finds excerpts that might carry a rare dimension; it cannot tell which
actually do, which is why the round-3 candidates were only hints. This runs the real
classifier over the pool so the next coding batch can be selected on the model's
judgement instead of on a regex.

Uses the fold-0 models. Those were trained on four fifths of the labelled data, and the
pool is unlabelled corpus text they have never seen, so there is no leakage concern.

Batched generation with left padding: the system prompt is a shared prefix, so batching
is worth about six times the sequential rate measured on this hardware (1.27 -> 7.99
excerpts/s/GPU at batch 64 on a 48GB card).

    python code/40.score_pool.py                       # full pool
    ML_POOL_LIMIT=2000 python code/40.score_pool.py    # smoke
    ML_SCORE_BATCH=16 python code/40.score_pool.py     # 32GB cards
"""
import os, sys, time, gc

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multilabel_prompt import (MODEL_SLUG, MAX_SEQ_LEN as GATE_SEQ,
                               format_prompt as gate_prompt, im_end_id)
from metrics_multilabel import parse_two_digit
from hier_prompt import (MAX_SEQ_LEN as HIER_SEQ, dims_for, n_digits,
                         format_prompt as hier_prompt, parse_digits)
from unsloth import FastModel

POOL = "data/trainingfinal/round4_pool.csv"
OUT = "data/trainingfinal/round4_scored.csv"
FOLD = int(os.environ.get("ML_SCORE_FOLD", 0))
BATCH = int(os.environ.get("ML_SCORE_BATCH", 48))
LIMIT = int(os.environ.get("ML_POOL_LIMIT", 0))
SC_DIMS, JC_DIMS = dims_for("sc"), dims_for("jc")


def load(path, seq):
    m, t = FastModel.from_pretrained(model_name=path, max_seq_length=seq, load_in_4bit=True,
                                     load_in_16bit=False, full_finetuning=False,
                                     device_map="balanced")
    if hasattr(t, "tokenizer"):
        t = t.tokenizer
    m.eval()
    t.padding_side = "left"
    if t.pad_token_id is None:
        t.pad_token = t.eos_token
    return m, t


def run_batched(model, tok, prompts, max_new, label):
    """Greedy-decode a list of prompts in batches; return raw completions in order."""
    out, eos = [], im_end_id(tok)
    t0, n = time.time(), len(prompts)
    for i in range(0, n, BATCH):
        enc = tok(prompts[i:i + BATCH], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                               eos_token_id=eos, pad_token_id=tok.eos_token_id)
        out += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        del enc, g
        done = min(i + BATCH, n)
        if done % (BATCH * 20) < BATCH or done == n:
            el = time.time() - t0
            print(f"    {label}: {done}/{n}  {done/el:.1f} rows/s  "
                  f"eta {(n-done)/(done/el)/60:.0f} min", flush=True)
    return out


def main():
    df = pd.read_csv(POOL)
    if LIMIT:
        df = df.head(LIMIT).copy()
    df = df.reset_index(drop=True)
    print(f"pool: {len(df)} rows from {POOL}", flush=True)

    # ---- stage 1: the gate ----
    p = f"models/finetuned/{MODEL_SLUG}_multilabel/outer{FOLD}"
    print(f"loading gate {p}", flush=True)
    m, t = load(p, GATE_SEQ)
    raw = run_batched(m, t, [gate_prompt(x) for x in df["review_text"].astype(str)], 8, "gate")
    parsed = [parse_two_digit(r) for r in raw]
    df["pred_schedule"] = [a for a, b, c in parsed]
    df["pred_jobcontrol"] = [b for a, b, c in parsed]
    df["gate_ok"] = [int(c) for a, b, c in parsed]
    del m, t
    gc.collect(); torch.cuda.empty_cache()
    print(f"  gate: schedule {df.pred_schedule.sum()}  job-control {df.pred_jobcontrol.sum()}"
          f"  malformed {len(df)-df.gate_ok.sum()}", flush=True)

    # ---- stage 2: dimensions, only where the gate opened ----
    for branch, dims, col in (("sc", SC_DIMS, "pred_schedule"), ("jc", JC_DIMS, "pred_jobcontrol")):
        for d in dims:
            df[d] = 0
        sel = df.index[df[col] == 1].tolist()
        print(f"branch {branch}: {len(sel)} rows routed", flush=True)
        if not sel:
            continue
        p = f"models/finetuned/{MODEL_SLUG}_hier_{branch}/outer{FOLD}"
        print(f"  loading {p}", flush=True)
        m, t = load(p, HIER_SEQ)
        prompts = [hier_prompt(branch, df.at[i, "review_text"], df.at[i, "type"]) for i in sel]
        raw = run_batched(m, t, prompts, n_digits(branch) + 6, branch)
        for i, r in zip(sel, raw):
            digits, ok = parse_digits(r, branch)
            for d, v in zip(dims, digits):
                df.at[i, d] = int(v)
        del m, t
        gc.collect(); torch.cuda.empty_cache()

    df["n_pred_dims"] = df[SC_DIMS + JC_DIMS].sum(axis=1)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(df)} rows)", flush=True)
    print("\npredicted positives per dimension:")
    for d in SC_DIMS + JC_DIMS:
        print(f"  {int(df[d].sum()):>5}  {d}")


if __name__ == "__main__":
    main()
