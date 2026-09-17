"""
Stage 2 of the hierarchical classifier: dimension-level labels within each branch.

Stage 1 (11.finetune_multilabel.py) decides schedule_related / job_control_related.
Stage 2 takes a review already known to belong to a branch and decides WHICH dimensions
of that branch apply.

Two branches, each its own model:
  sc : schedule dimensions, 20 positions (10 cons then 10 pros)
  jc : job-control dimensions, 7 positions (5 cons then 2 pros)

Folds are INHERITED from data/trainingfinal/labelled_cv.csv rather than re-drawn, so a
review held out at stage 1 is the same review held out at stage 2. That makes the
end-to-end (gate then dimensions) evaluation honest: no stage-2 model has ever seen a
row that its own stage-1 gate is being scored on.

Input : data/trainingfinal/labelled_dimensions.csv  (id + one 0/1 column per dimension)
        data/trainingfinal/labelled_cv.csv          (id, fold, review_text, type)
Output: data/trainingfinal/hier_{sc,jc}_cv.csv

Run from the repo root:
    python code/20.prepare_hierarchical_data.py
"""
import sys
import pandas as pd

CV_CSV   = "data/trainingfinal/labelled_cv.csv"
DIM_CSV  = "data/trainingfinal/labelled_dimensions.csv"

SC_CONS = ["long hour and overwork",
           "inflexible schedule or no autonomy in scheduling",
           "schedule and hour variability",
           "insufficient hours",
           "low predictability and short notice",
           "undesirable hours or days",
           "lack of paid time off (PTO), paid vacation, or paid sick leave",
           "broad life-work conflict",
           "lack of work-from-home or remote work opportunities",
           "general complaints"]
SC_PROS = ["regular hours or no overwork",
           "flexibility or autonomy in scheduling",
           "desirable hours or days",
           "hour consistency",
           "hour predictability or advance notice",
           "paid time off (PTO), paid vacation, or paid sick leave",
           "additional hours opportunities",
           "work-life balance",
           "work-from-home or remote work opportunities",
           "overall compliment"]
JC_CONS = ["under-staffing and unmanageable workload",
           "over-staffing or nothing to do",
           "lack of job autonomy and flexibility in general",
           "uneven or unfair allocation of work",
           "inconsistent or cyclical workload over time"]
JC_PROS = ["manageable workload or adequate staffing",
           "job autonomy and flexibility in general"]

BRANCHES = {
    "sc": {"gate": "schedule_related",    "dims": SC_CONS + SC_PROS, "n_cons": len(SC_CONS)},
    "jc": {"gate": "job_control_related", "dims": JC_CONS + JC_PROS, "n_cons": len(JC_CONS)},
}


def main():
    cv = pd.read_csv(CV_CSV)
    dims = pd.read_csv(DIM_CSV)
    missing = set(cv["id"]) - set(dims["id"])
    if missing:
        raise SystemExit(f"{len(missing)} id(s) in {CV_CSV} absent from {DIM_CSV}: {sorted(missing)[:10]}")

    df = cv.merge(dims, on="id", how="left", suffixes=("", "_dim"))

    for tag, spec in BRANCHES.items():
        cols = spec["dims"]
        for c in cols:
            if c not in df.columns:
                raise SystemExit(f"dimension column missing from {DIM_CSV}: {c!r}")
        sub = df[df[spec["gate"]] == 1].copy().reset_index(drop=True)
        sub[cols] = sub[cols].fillna(0).astype(int)

        # Every row in a branch must carry at least one of that branch's dimensions,
        # otherwise the gate and the dimension coding disagree.
        empty = sub[sub[cols].sum(axis=1) == 0]
        if len(empty):
            raise SystemExit(f"[{tag}] {len(empty)} gated row(s) with no dimension set: "
                             f"{empty['id'].tolist()[:10]}")

        # Polarity sanity: a cons excerpt should not light a pros dimension and vice versa.
        n_cons = spec["n_cons"]
        cross_c = sub[(sub["type"] == "c") & (sub[cols[n_cons:]].sum(axis=1) > 0)]
        cross_p = sub[(sub["type"] == "p") & (sub[cols[:n_cons]].sum(axis=1) > 0)]
        print(f"[{tag}] cross-polarity rows: cons-with-pro-dim={len(cross_c)}, "
              f"pros-with-con-dim={len(cross_p)} (expected: a few, the cross-polarity rule)")

        sub["target"] = sub[cols].astype(str).agg("".join, axis=1)
        out = sub[["id", "source", "type", "review_text", "fold", "target"] + cols]
        path = f"data/trainingfinal/hier_{tag}_cv.csv"
        out.to_csv(path, index=False)

        print(f"\n=== branch {tag}: {len(out)} rows -> {path} ===")
        print(f"  digits per row: {len(cols)}  (cons 1..{n_cons}, pros {n_cons+1}..{len(cols)})")
        print(f"  mean dimensions per row: {sub[cols].sum(axis=1).mean():.2f}")
        print(f"  fold sizes: {out['fold'].value_counts().sort_index().to_dict()}")
        freq = sub[cols].sum().sort_values(ascending=False)
        print("  positives per dimension:")
        for name, n in freq.items():
            flag = "   <-- too rare to learn" if n < 20 else ("   <-- thin" if n < 40 else "")
            print(f"    {n:>4}  {name}{flag}")


if __name__ == "__main__":
    sys.exit(main())
