"""
Aggregate whatever the hierarchy has finished so far into one readable report.

Reads every cv_results_*.json written by 11.finetune_multilabel.py (stage 1) and
21.finetune_hierarchical.py (stage 2) and prints:
  - stage 1: schedule / job-control F1 per outer fold and pooled
  - stage 2: per-dimension F1 and training support, per branch
  - a labelling verdict per dimension: whether its support is the binding constraint

Safe to run at any time; folds still training are simply listed as pending.

    python code/25.report.py                 # text
    python code/25.report.py --csv out.csv   # also write a per-dimension table
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

SLUG = "Qwen3.5-35B-A3B"


def load(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        try:
            with open(p) as f:
                out.append((p, json.load(f)))
        except Exception as e:
            print(f"  (skipping {p}: {e})")
    return out


def stage1():
    rows = load(f"models/cv_results_multilabel/{SLUG}/outer*/cv_results_*.json")
    print("=" * 78)
    print("STAGE 1  gate: schedule_related / job_control_related")
    print("=" * 78)
    if not rows:
        print("  no completed folds yet.\n")
        return None
    recs = []
    for p, r in rows:
        recs.append({"outer_fold": r["outer_fold"], "best_trial": r["best_trial"],
                     "macro_f1": r.get("best_mean_macro_f1"),
                     "schedule_f1": r.get("best_mean_schedule_f1"),
                     "params": r["best_params"], "n_train": r["n_outer_train"],
                     "n_test": r["n_outer_test"]})
    df = pd.DataFrame(recs).sort_values("outer_fold")
    for _, x in df.iterrows():
        print(f"  fold {int(x.outer_fold)}  macro_f1={x.macro_f1 if x.macro_f1 is None else round(x.macro_f1,4)}"
              f"  n_train={x.n_train} n_test={x.n_test}  best={x.params}")
    m = df["macro_f1"].dropna()
    if len(m):
        print(f"\n  pooled over {len(m)} fold(s): macro_f1 mean={m.mean():.4f} sd={m.std(ddof=0):.4f}")
    print(f"  folds finished: {len(df)}/5\n")
    return df


def stage2(branch, label):
    rows = load(f"models/cv_results_hier/{SLUG}/{branch}/outer*/cv_results_*.json")
    print("=" * 78)
    print(f"STAGE 2  branch {branch}: {label}")
    print("=" * 78)
    if not rows:
        print("  no completed folds yet.\n")
        return None
    dims = rows[0][1]["dimensions"]
    per_fold, micro, macro, support = {}, [], [], {}
    for p, r in rows:
        k = r["outer_fold"]
        micro.append(r.get("best_mean_micro_f1"))
        macro.append(r.get("best_mean_macro_f1"))
        pd_ = r.get("best_mean_per_dim_f1") or {}
        per_fold[k] = pd_
        for d, n in (r.get("train_pool_support") or {}).items():
            support[d] = max(support.get(d, 0), int(n))
    print(f"  folds finished: {len(rows)}/5")
    mi = [x for x in micro if x is not None]
    ma = [x for x in macro if x is not None]
    if mi:
        print(f"  micro_f1 mean={np.mean(mi):.4f}   macro_f1 mean={np.mean(ma):.4f}\n")
    recs = []
    for d in dims:
        vals = [per_fold[k][d] for k in per_fold if d in per_fold[k]]
        recs.append({"branch": branch, "dimension": d, "support_train": support.get(d, 0),
                     "f1_mean": float(np.mean(vals)) if vals else np.nan,
                     "f1_sd": float(np.std(vals, ddof=0)) if len(vals) > 1 else np.nan,
                     "n_folds": len(vals)})
    t = pd.DataFrame(recs).sort_values("f1_mean", ascending=False, na_position="last")
    print(f"  {'F1':>6}  {'sd':>5}  {'n+':>5}   dimension")
    for _, x in t.iterrows():
        f1 = "  n/a" if np.isnan(x.f1_mean) else f"{x.f1_mean:.3f}"
        sd = "     " if np.isnan(x.f1_sd) else f"{x.f1_sd:.3f}"
        print(f"  {f1:>6}  {sd:>5}  {int(x.support_train):>5}   {x.dimension}")
    print()
    return t


def verdict(t):
    """Label each dimension by whether more hand-coding is the binding constraint."""
    if t is None or not len(t):
        return t
    def v(r):
        n, f1 = r.support_train, r.f1_mean
        if np.isnan(f1):
            return "not yet evaluated"
        if n < 20:
            return "MORE LABELLING NEEDED (support < 20; F1 not trustworthy)"
        if f1 >= 0.75:
            return "usable as is"
        if f1 >= 0.55:
            return "borderline; more labelling likely helps"
        if n < 60:
            return "MORE LABELLING NEEDED (low F1 at low support)"
        return "weak despite support; likely a definition problem, not a data problem"
    t = t.copy()
    t["verdict"] = t.apply(v, axis=1)
    return t


def human_ceiling():
    """What two humans actually agree on, as the yardstick for the model numbers."""
    path = "data/trainingfinal/human_ceiling_2026-09-12.csv"
    if not os.path.exists(path):
        return
    h = pd.read_csv(path)
    print("=" * 78)
    print("HUMAN CEILING  (raw pairwise agreement between coders, before adjudication)")
    print("=" * 78)
    for _, r in h.iterrows():
        print(f"  {r['pair']:<26} n={int(r['n']):>4}  schedule F1 {r['sched_f1']:.3f}   "
              f"job-control F1 {r['jc_f1']:.3f}   macro {r['macro_f1']:.3f}")
    print("  Model numbers below are scored against ADJUDICATED gold, so they are not a")
    print("  like-for-like comparison, but they bound what agreement on this task looks like.\n")


def boot_ci(y, p, n_boot=2000, seed=0):
    """Bootstrap 95% CI for F1, so small-support dimensions carry visible uncertainty."""
    from sklearn.metrics import f1_score as _f1
    rng = np.random.RandomState(seed)
    n = len(y)
    if n == 0 or y.sum() == 0:
        return (np.nan, np.nan)
    vals = []
    for _ in range(n_boot):
        i = rng.randint(0, n, n)
        vals.append(_f1(y[i], p[i], zero_division=0))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    human_ceiling()
    stage1()
    sc = verdict(stage2("sc", "20 schedule dimensions"))
    jc = verdict(stage2("jc", "7 job-control dimensions"))

    tabs = [x for x in (sc, jc) if x is not None and len(x)]
    if tabs:
        allt = pd.concat(tabs, ignore_index=True)
        print("=" * 78)
        print("LABELLING VERDICT BY DIMENSION")
        print("=" * 78)
        for verd, grp in allt.groupby("verdict"):
            print(f"\n{verd}  ({len(grp)})")
            for _, x in grp.sort_values("support_train").iterrows():
                f1 = "n/a" if np.isnan(x.f1_mean) else f"{x.f1_mean:.3f}"
                print(f"   n={int(x.support_train):>4}  F1={f1:>5}   {x.dimension}")
        if a.csv:
            allt.to_csv(a.csv, index=False)
            print(f"\nwrote {a.csv}")
    print()


if __name__ == "__main__":
    main()
