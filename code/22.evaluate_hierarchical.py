"""
Score stage 2 on the held-out outer folds and pool across all five.

The per-dimension numbers written during training come from the INNER cross-validation,
where each validation slice is only about a third of one fold's training pool. For a rare
dimension that is two or three positives, which is not a number to make decisions on.

This script instead loads each outer fold's final adapter, predicts that fold's held-out
rows, and pools the five folds. Every labelled row is then predicted exactly once by a
model that never saw it, so per-dimension F1 rests on the dimension's FULL support
(794 rows for sc, 607 for jc) rather than a fifth of it.

That is the number to use when deciding whether a dimension needs more hand-coding.

    ML_BRANCH=sc python code/22.evaluate_hierarchical.py
"""
import os
import sys
import json
import gc

BRANCH = os.environ.get("ML_BRANCH", "sc")
if BRANCH not in ("sc", "jc"):
    raise SystemExit("ML_BRANCH must be sc or jc")

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hier_prompt import (MODEL_NAME, MODEL_SLUG, MAX_SEQ_LEN, dims_for, n_digits,
                         parse_digits, generate_digits)
from unsloth import FastModel

DIMS = dims_for(BRANCH)
NDIG = n_digits(BRANCH)
DATA_CSV = f"data/trainingfinal/hier_{BRANCH}_cv.csv"
OUT_DIR = f"models/eval_hier/{MODEL_SLUG}/{BRANCH}"
os.makedirs(OUT_DIR, exist_ok=True)


def load_adapter(fold):
    d = f"models/finetuned/{MODEL_SLUG}_hier_{BRANCH}/outer{fold}"
    if not os.path.isdir(d):
        return None, None, d
    model, tok = FastModel.from_pretrained(
        model_name=d, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True,
        load_in_16bit=False, full_finetuning=False, device_map="balanced")
    if hasattr(tok, "tokenizer"):
        tok = tok.tokenizer
    model.eval()
    return model, tok, d


def main():
    df = pd.read_csv(DATA_CSV, dtype={"target": str})
    df["target"] = df["target"].str.zfill(NDIG)

    frames, missing = [], []
    for fold in range(5):
        test = df[df["fold"] == fold].reset_index(drop=True)
        model, tok, path = load_adapter(fold)
        if model is None:
            missing.append(fold)
            print(f"fold {fold}: no final adapter at {path}; skipping")
            continue
        print(f"fold {fold}: scoring {len(test)} held-out rows with {path}")
        preds, raws, malformed = [], [], 0
        for _, row in test.iterrows():
            t = generate_digits(model, tok, BRANCH, row["review_text"], row["type"])
            d, ok = parse_digits(t, BRANCH)
            malformed += (not ok)
            preds.append(d)
            raws.append(t)
        out = test[["id", "source", "type", "review_text", "target"]].copy()
        out["fold"] = fold
        out["pred"] = ["".join(map(str, p)) for p in preds]
        out["raw"] = raws
        frames.append(out)
        print(f"  malformed: {malformed}/{len(test)}")
        del model, tok
        gc.collect(); torch.cuda.empty_cache()

    if not frames:
        raise SystemExit("No final adapters found. Run the training first.")

    pooled = pd.concat(frames, ignore_index=True)
    pooled.to_csv(os.path.join(OUT_DIR, f"oof_predictions_{BRANCH}.csv"), index=False)

    y = np.array([[int(c) for c in t] for t in pooled["target"]])
    p = np.array([[int(c) for c in t] for t in pooled["pred"]])

    rows = []
    for j, d in enumerate(DIMS):
        rows.append({
            "dimension": d, "support": int(y[:, j].sum()),
            "precision": float(precision_score(y[:, j], p[:, j], zero_division=0)),
            "recall": float(recall_score(y[:, j], p[:, j], zero_division=0)),
            "f1": float(f1_score(y[:, j], p[:, j], zero_division=0)),
        })
    t = pd.DataFrame(rows).sort_values("f1", ascending=False)
    micro = float(f1_score(y.ravel(), p.ravel(), zero_division=0))
    macro = float(t.loc[t["support"] > 0, "f1"].mean())
    exact = float((y == p).all(axis=1).mean())

    print(f"\n{'='*78}\nPOOLED OUT-OF-FOLD, branch {BRANCH}: "
          f"{len(pooled)} rows, {len(missing)} fold(s) missing {missing}\n{'='*78}")
    print(f"  micro F1 {micro:.4f}   macro F1 {macro:.4f}   exact-match {exact:.4f}\n")
    print(f"  {'F1':>6} {'prec':>6} {'rec':>6} {'n+':>5}   dimension")
    for _, r in t.iterrows():
        print(f"  {r.f1:6.3f} {r.precision:6.3f} {r.recall:6.3f} {int(r.support):5d}   {r.dimension}")

    t.to_csv(os.path.join(OUT_DIR, f"per_dimension_{BRANCH}.csv"), index=False)
    json.dump({"branch": BRANCH, "n_rows": len(pooled), "folds_missing": missing,
               "micro_f1": micro, "macro_f1": macro, "exact_match": exact,
               "per_dimension": rows},
              open(os.path.join(OUT_DIR, f"pooled_{BRANCH}.json"), "w"), indent=2)
    print(f"\nwrote {OUT_DIR}/per_dimension_{BRANCH}.csv")


if __name__ == "__main__":
    main()
