"""
End-to-end evaluation of the two-stage classifier, the way it will actually be used.

Stage 2 is trained and scored on rows the GOLD labels say belong to a branch. In
production nothing hands it gold: stage 1 decides the gate, and a review the gate wrongly
rejects never reaches stage 2 at all. Errors therefore compound, and the honest number for
the 3.55M-review run is this one, not the gold-gated stage-2 number.

For each outer fold K this loads that fold's three final models, runs the real pipeline on
fold K's held-out rows, and pools all five folds so every labelled row is predicted exactly
once by models that never saw it.

Reports, per dimension: end-to-end F1, the gold-gated F1 for comparison, and the share of
the loss attributable to the gate.

    python code/23.evaluate_pipeline.py
"""
import os
import sys
import json
import gc

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multilabel_prompt import MODEL_SLUG, generate_two_digit
from metrics_multilabel import parse_two_digit
from hier_prompt import dims_for, n_digits, parse_digits, generate_digits
from hier_prompt import MAX_SEQ_LEN as HIER_SEQ
from multilabel_prompt import MAX_SEQ_LEN as GATE_SEQ
from unsloth import FastModel

SC_DIMS, JC_DIMS = dims_for("sc"), dims_for("jc")
ALL_DIMS = SC_DIMS + JC_DIMS
OUT_DIR = f"models/eval_pipeline/{MODEL_SLUG}"
os.makedirs(OUT_DIR, exist_ok=True)


def load(path, seq):
    if not os.path.isdir(path):
        return None, None
    m, t = FastModel.from_pretrained(model_name=path, max_seq_length=seq, load_in_4bit=True,
                                     load_in_16bit=False, full_finetuning=False,
                                     device_map="balanced")
    if hasattr(t, "tokenizer"):
        t = t.tokenizer
    m.eval()
    return m, t


def free(m, t):
    del m, t
    gc.collect(); torch.cuda.empty_cache()


def main():
    gate = pd.read_csv("data/trainingfinal/labelled_cv.csv")
    sc = pd.read_csv("data/trainingfinal/hier_sc_cv.csv", dtype={"target": str})
    jc = pd.read_csv("data/trainingfinal/hier_jc_cv.csv", dtype={"target": str})
    sc["target"] = sc["target"].str.zfill(n_digits("sc"))
    jc["target"] = jc["target"].str.zfill(n_digits("jc"))

    # gold 27-dimension matrix for every labelled row
    gold = gate[["id", "type", "review_text", "fold",
                 "schedule_related", "job_control_related"]].copy()
    for d in SC_DIMS:
        gold[d] = gold["id"].map(sc.set_index("id")[d]).fillna(0).astype(int)
    for d in JC_DIMS:
        gold[d] = gold["id"].map(jc.set_index("id")[d]).fillna(0).astype(int)

    rows = []
    for fold in range(5):
        test = gold[gold["fold"] == fold].reset_index(drop=True)
        p_gate = f"models/finetuned/{MODEL_SLUG}_multilabel/outer{fold}"
        p_sc = f"models/finetuned/{MODEL_SLUG}_hier_sc/outer{fold}"
        p_jc = f"models/finetuned/{MODEL_SLUG}_hier_jc/outer{fold}"
        if not all(os.path.isdir(p) for p in (p_gate, p_sc, p_jc)):
            print(f"fold {fold}: missing one of the three models; skipping")
            continue

        print(f"fold {fold}: gating {len(test)} held-out rows")
        m, t = load(p_gate, GATE_SEQ)
        gs, gj = [], []
        for _, r in test.iterrows():
            s, c, _ = parse_two_digit(generate_two_digit(m, t, r["review_text"], max_new_tokens=8))
            gs.append(s); gj.append(c)
        free(m, t)
        test["pred_sched"], test["pred_jc"] = gs, gj

        pred = pd.DataFrame(0, index=test.index, columns=ALL_DIMS)
        for branch, dims, path in (("sc", SC_DIMS, p_sc), ("jc", JC_DIMS, p_jc)):
            sel = test.index[test["pred_sched" if branch == "sc" else "pred_jc"] == 1]
            print(f"  stage 2 {branch}: gate routed {len(sel)} rows")
            if not len(sel):
                continue
            m, t = load(path, HIER_SEQ)
            for i in sel:
                d, _ = parse_digits(generate_digits(m, t, branch, test.at[i, "review_text"],
                                                    test.at[i, "type"]), branch)
                pred.loc[i, dims] = d
            free(m, t)

        out = test[["id", "type", "fold", "schedule_related", "job_control_related",
                    "pred_sched", "pred_jc"] + ALL_DIMS].copy()
        for d in ALL_DIMS:
            out[f"pred__{d}"] = pred[d].values
        rows.append(out)

    if not rows:
        raise SystemExit("No fold had all three models. Run the training first.")

    pooled = pd.concat(rows, ignore_index=True)
    pooled.to_csv(os.path.join(OUT_DIR, "pipeline_oof_predictions.csv"), index=False)

    y = pooled[ALL_DIMS].to_numpy()
    p = pooled[[f"pred__{d}" for d in ALL_DIMS]].to_numpy()

    recs = []
    for j, d in enumerate(ALL_DIMS):
        recs.append({"dimension": d, "branch": "sc" if d in SC_DIMS else "jc",
                     "support": int(y[:, j].sum()),
                     "precision": float(precision_score(y[:, j], p[:, j], zero_division=0)),
                     "recall": float(recall_score(y[:, j], p[:, j], zero_division=0)),
                     "f1_end_to_end": float(f1_score(y[:, j], p[:, j], zero_division=0))})
    t = pd.DataFrame(recs)

    gate_s_f1 = f1_score(pooled["schedule_related"], pooled["pred_sched"], zero_division=0)
    gate_j_f1 = f1_score(pooled["job_control_related"], pooled["pred_jc"], zero_division=0)
    micro = f1_score(y.ravel(), p.ravel(), zero_division=0)
    macro = t.loc[t["support"] > 0, "f1_end_to_end"].mean()

    print(f"\n{'='*78}\nEND-TO-END PIPELINE, pooled over {pooled['fold'].nunique()} fold(s), "
          f"{len(pooled)} rows\n{'='*78}")
    print(f"  gate: schedule F1 {gate_s_f1:.4f}   job-control F1 {gate_j_f1:.4f}")
    print(f"  dimensions: micro F1 {micro:.4f}   macro F1 {macro:.4f}\n")
    print(f"  {'F1':>6} {'prec':>6} {'rec':>6} {'n+':>5}  br  dimension")
    for _, r in t.sort_values("f1_end_to_end", ascending=False).iterrows():
        print(f"  {r.f1_end_to_end:6.3f} {r.precision:6.3f} {r.recall:6.3f} "
              f"{int(r.support):5d}  {r.branch}  {r.dimension}")

    t.to_csv(os.path.join(OUT_DIR, "per_dimension_end_to_end.csv"), index=False)
    json.dump({"n_rows": int(len(pooled)), "folds": int(pooled["fold"].nunique()),
               "gate_schedule_f1": float(gate_s_f1), "gate_jobcontrol_f1": float(gate_j_f1),
               "dimension_micro_f1": float(micro), "dimension_macro_f1": float(macro),
               "per_dimension": recs},
              open(os.path.join(OUT_DIR, "pipeline_summary.json"), "w"), indent=2)
    print(f"\nwrote {OUT_DIR}/per_dimension_end_to_end.csv")


if __name__ == "__main__":
    main()
