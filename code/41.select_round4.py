"""
Turn the scored candidate pool into per-coder coding assignments.

The budget is coder-slots, not excerpts: with two coders at 500 each, 1,000 slots buys
1,000 distinct excerpts when there is no double coding, or fewer if an overlap block is
kept for an inter-coder agreement estimate. Set `overlap` in round4_plan.json.

Three kinds of row go in, deliberately:

  confirmed      the classifier says the dimension applies; these fill the support gap.
                 Allocated per dimension as gap / F1, so a dimension the model is shakier
                 on gets proportionally more candidates to survive its own false positives.
  contradiction  the keyword search pulled it FOR that dimension but the classifier says
                 no. These sit on the boundary the model currently gets wrong, so they
                 teach more per coded row than a confirmation does.
  anchor         random corpus excerpts, never keyword-matched. Without these the batch
                 only ever confirms what the model already believes, and the next training
                 round inherits its blind spots.

Reads round4_scored.csv and round4_plan.json; writes round4_selected.csv plus one CSV
per coder.
"""
import json
import numpy as np
import pandas as pd

SCORED = "data/trainingfinal/round4_scored.csv"
PLAN = "data/trainingfinal/round4_plan.json"
CODERS = ["WJ", "LM"]
SEED = 42


def main():
    df = pd.read_csv(SCORED)
    plan = json.load(open(PLAN))
    alloc, n_contra, n_anchor, n_overlap = (plan["alloc"], plan["contra_total"],
                                            plan["anchor"], plan["overlap"])
    rng = np.random.RandomState(SEED)
    chosen, why = {}, {}

    def take(idx, reason, dim, k):
        idx = [i for i in idx if i not in chosen]
        rng.shuffle(idx)
        for i in idx[:k]:
            chosen[i], why[i] = dim, reason
        return min(k, len(idx))

    # rarest dimensions first, so scarce candidates land where they are needed most
    for dim, k in sorted(alloc.items(), key=lambda kv: -kv[1]):
        if dim not in df.columns:
            print(f"  ! {dim} missing from scored file"); continue
        got = take(df.index[df[dim] == 1].tolist(), "confirmed", dim, k)
        if got < k:
            print(f"  ! only {got}/{k} confirmed candidates for {dim}")

    per_dim_contra = max(1, n_contra // max(len(alloc), 1))
    for dim in alloc:
        take(df.index[(df["mined_for"] == dim) & (df.get(dim, 0) == 0)].tolist(),
             "contradiction", dim, per_dim_contra)
    take(df.index[df["mined_for"] == "(random sample)"].tolist(),
         "anchor", "(none targeted)", n_anchor)

    sel = df.loc[sorted(chosen)].copy()
    sel["selected_for"] = [chosen[i] for i in sel.index]
    sel["why"] = [why[i] for i in sel.index]
    sel = sel.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # shared block stratified across the three row kinds, so reliability is measured on
    # the same mix everyone codes rather than only on the easy confirmations
    shared_idx = []
    for reason, share in (("confirmed", 0.70), ("contradiction", 0.18), ("anchor", 0.12)):
        pool = sel.index[(sel["why"] == reason) & (~sel.index.isin(shared_idx))].tolist()
        rng.shuffle(pool)
        shared_idx += pool[:int(round(n_overlap * share))]
    sel["shared"] = sel.index.isin(shared_idx).astype(int)

    rest = sel.index[sel["shared"] == 0].tolist()
    rng.shuffle(rest)
    per = int(np.ceil(len(rest) / len(CODERS)))
    assign = {}
    for c, start in zip(CODERS, range(0, len(rest), per)):
        for i in rest[start:start + per]:
            assign[i] = c
    sel["assigned"] = [assign.get(i, "") for i in sel.index]
    sel.to_csv("data/trainingfinal/round4_selected.csv", index=False)

    print(f"\nselected {len(sel)} distinct excerpts")
    print(sel["why"].value_counts().to_string())
    print(f"\nshared (every coder codes these): {int(sel['shared'].sum())}")
    for c in CODERS:
        sub = sel[(sel["assigned"] == c) | (sel["shared"] == 1)]
        sub.to_csv(f"data/trainingfinal/round4_assign_{c}.csv", index=False)
        print(f"  {c}: {len(sub)} rows  -> data/trainingfinal/round4_assign_{c}.csv")
    print("\nrows per targeted dimension:")
    t = sel.groupby(["selected_for", "why"]).size().unstack(fill_value=0)
    t["total"] = t.sum(axis=1)
    print(t.sort_values("total", ascending=False).to_string())


if __name__ == "__main__":
    main()
