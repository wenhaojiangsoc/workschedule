"""
Pool the five outer folds into one out-of-fold estimate.

11.finetune_multilabel.py trains one model per outer fold; 12.evaluate_multilabel.py
scores each on its own held-out fold. This script joins those five prediction files into a
single out-of-fold (OOF) prediction set covering every evaluable row exactly once, and
reports:

  1. The pooled OOF multi-label report -- the headline number. Because it spans the whole
     gold set (~1559 rows) rather than one 190-row slice, the bootstrap CIs in
     metrics_multilabel finally have enough positives to say something about job_control.
  2. A per-fold table with mean +/- sd. The sd is the honest measure of how much any single
     split's number was luck.
  3. Hyperparameter stability across the five independent Optuna searches. Under nested CV
     the searches never share information, so whether they converge on similar r / lr is
     itself a result worth reporting.

CPU-only -- no GPU, no unsloth. Run from the repo root:
    python code/15.aggregate_multilabel_cv.py
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_multilabel import multilabel_report
from multilabel_prompt import MODEL_SLUG

DATA_CSV    = "data/trainingfinal/labelled_cv.csv"
PREDS_GLOB  = f"data/trainingfinal/val_preds_{MODEL_SLUG}_outer*.csv"
RESULTS_DIR = f"models/cv_results_multilabel/{MODEL_SLUG}"
OOF_CSV     = f"data/trainingfinal/oof_preds_{MODEL_SLUG}.csv"
SUMMARY_JSON = "outputs/multilabel_cv_summary.json"
TARGET_F1   = 0.95


def load_oof(data_csv, preds_glob):
    """Concatenate the per-fold prediction files and check they partition the data."""
    gold = pd.read_csv(data_csv)
    evaluable = gold[gold["fold"] >= 0]
    expected_folds = sorted(evaluable["fold"].unique())

    paths = sorted(glob.glob(preds_glob))
    if not paths:
        raise SystemExit(
            f"No prediction files matched {preds_glob}.\n"
            f"  Run 12.evaluate_multilabel.py for each fold first "
            f"(sbatch code/SBATCH/12.evaluate_multilabel.sh)."
        )

    frames = []
    for p in paths:
        d = pd.read_csv(p)
        if "fold" not in d.columns:
            raise SystemExit(f"{p} has no `fold` column; re-run 12.evaluate_multilabel.py.")
        if d["fold"].nunique() != 1:
            raise SystemExit(f"{p} mixes folds {sorted(d['fold'].unique())}")
        print(f"  {os.path.basename(p):<50} fold {int(d['fold'].iloc[0])}, {len(d)} rows")
        frames.append(d)
    preds = pd.concat(frames, ignore_index=True)

    got_folds = sorted(preds["fold"].unique())
    if got_folds != expected_folds:
        raise SystemExit(
            f"Fold coverage mismatch: {data_csv} defines folds {expected_folds}, "
            f"predictions cover {got_folds}. Missing folds would silently bias the pooled "
            f"metric toward whichever folds did finish."
        )

    # The real test that the outer folds partition the data: exactly one prediction per row.
    dupes = preds["id"].duplicated().sum()
    if dupes:
        raise SystemExit(f"{dupes} id(s) predicted more than once -- folds overlap.")
    missing = set(evaluable["id"]) - set(preds["id"])
    extra = set(preds["id"]) - set(evaluable["id"])
    if missing:
        raise SystemExit(f"{len(missing)} evaluable row(s) have no prediction, "
                         f"e.g. {sorted(missing)[:10]}")
    if extra:
        raise SystemExit(f"{len(extra)} predicted id(s) are not evaluable rows "
                         f"(pinned few-shot leaked into a test fold?): {sorted(extra)[:10]}")

    merged = evaluable.merge(
        preds[["id", "fold", "sched_pred", "ctrl_pred"]].rename(columns={"fold": "pred_fold"}),
        on="id", how="left", validate="one_to_one",
    )
    if (merged["fold"] != merged["pred_fold"]).any():
        raise SystemExit("A row was scored by a model that was trained on it. Check the "
                         "fold assignment in 10.prepare_multilabel_data.py.")
    return merged.drop(columns=["pred_fold"])


def per_fold_table(oof):
    """Per-fold F1s computed from the pooled frame, so one code path scores everything."""
    from sklearn.metrics import f1_score

    rows = []
    for k, grp in oof.groupby("fold"):
        s = f1_score(grp["schedule_related"], grp["sched_pred"], pos_label=1, zero_division=0)
        c = f1_score(grp["job_control_related"], grp["ctrl_pred"], pos_label=1, zero_division=0)
        rows.append({
            "fold": int(k), "n": len(grp),
            "schedule_f1": float(s), "job_control_f1": float(c),
            "macro_f1": float(0.5 * (s + c)),
            "exact_match": float(np.mean(
                (grp["sched_pred"] == grp["schedule_related"])
                & (grp["ctrl_pred"] == grp["job_control_related"])
            )),
        })
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def hyperparameter_stability(results_dir):
    """Winning config from each outer fold's independent Optuna search."""
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "outer*", "cv_results_*.json"))):
        if "_smoke" in os.path.basename(path):
            continue
        with open(path) as f:
            r = json.load(f)
        rows.append({
            "outer_fold": r.get("outer_fold"),
            "best_trial": r.get("best_trial"),
            **{k: v for k, v in (r.get("best_params") or {}).items()},
            "inner_mean_schedule_f1": r.get("best_mean_schedule_f1"),
            "inner_mean_macro_f1": r.get("best_mean_macro_f1"),
            "data_sha1": r.get("data_sha1"),
        })
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("outer_fold").reset_index(drop=True)

    shas = set(df["data_sha1"].dropna())
    if len(shas) > 1:
        print("\n  WARNING: the folds were trained against DIFFERENT versions of the data "
              f"(data_sha1 {sorted(shas)}). Re-run the folds whose hash is stale before "
              "reporting a pooled number.")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_CSV)
    ap.add_argument("--preds_glob", default=PREDS_GLOB)
    ap.add_argument("--results_dir", default=RESULTS_DIR)
    ap.add_argument("--oof_out", default=OOF_CSV)
    ap.add_argument("--summary_out", default=SUMMARY_JSON)
    args = ap.parse_args()

    print("Loading per-fold predictions...")
    oof = load_oof(args.data, args.preds_glob)
    print(f"\nOut-of-fold set: {len(oof)} rows, every evaluable row predicted exactly once.")

    # 1. pooled
    pooled = multilabel_report(
        oof["schedule_related"], oof["job_control_related"],
        oof["sched_pred"], oof["ctrl_pred"],
        name=f"{MODEL_SLUG} pooled out-of-fold ({oof['fold'].nunique()}-fold nested CV)",
        target_f1=TARGET_F1,
    )

    # 2. per fold
    table = per_fold_table(oof)
    print("\n" + "=" * 66)
    print("PER-FOLD BREAKDOWN")
    print("=" * 66)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    stats = {}
    for col in ("schedule_f1", "job_control_f1", "macro_f1", "exact_match"):
        m, sd = float(table[col].mean()), float(table[col].std(ddof=1))
        stats[col] = {"mean": m, "sd": sd, "min": float(table[col].min()),
                      "max": float(table[col].max())}
        print(f"  {col:<16} mean={m:.4f}  sd={sd:.4f}  "
              f"range=[{table[col].min():.4f}, {table[col].max():.4f}]")

    # 3. hyperparameter stability
    hp = hyperparameter_stability(args.results_dir)
    print("\n" + "=" * 66)
    print("HYPERPARAMETER STABILITY ACROSS THE FIVE INDEPENDENT SEARCHES")
    print("=" * 66)
    if hp is None:
        print(f"  No cv_results_*.json found under {args.results_dir}/outer*/ -- skipped.")
    else:
        print(hp.to_string(index=False))

    # by-source breakdown: did the round-2 expansion actually help where it was mined?
    if "source" in oof.columns:
        from sklearn.metrics import f1_score
        print("\nBy source (round 1 vs the round-2 job-control expansion):")
        for src, grp in oof.groupby("source"):
            s = f1_score(grp["schedule_related"], grp["sched_pred"], pos_label=1, zero_division=0)
            c = f1_score(grp["job_control_related"], grp["ctrl_pred"], pos_label=1, zero_division=0)
            print(f"  {src:<8} n={len(grp):<5} schedule_f1={s:.4f}  job_control_f1={c:.4f}")

    # --- write artifacts ---------------------------------------------
    oof.to_csv(args.oof_out, index=False)
    os.makedirs(os.path.dirname(args.summary_out) or ".", exist_ok=True)
    summary = {
        "model": MODEL_SLUG,
        "n_folds": int(oof["fold"].nunique()),
        "n_oof_rows": int(len(oof)),
        "pooled": pooled,
        "per_fold": table.to_dict(orient="records"),
        "per_fold_stats": stats,
        "hyperparameters": None if hp is None else hp.to_dict(orient="records"),
    }
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {args.oof_out} and {args.summary_out}")


if __name__ == "__main__":
    main()
