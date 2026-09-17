"""
Prepare the frozen multi-label cross-validation dataset.

Reads data/trainingfinal/labelled.csv -- the single hand-coded source of truth -- and
produces data/trainingfinal/labelled_cv.csv, the artifact consumed by
11.finetune_multilabel.py, 12.evaluate_multilabel.py and 15.aggregate_multilabel_cv.py.

labelled.csv itself still carries the legacy `strat`/`set` columns from the old single
75/25 split (kept alive only for 13.GPT_multilabel_benchmark.R). This script ignores both
and rebuilds an outer 5-fold partition instead: every labelled review is the test row for
exactly one of five models, so the reported metric covers the whole gold set instead of a
single 190-row slice. dataset_three_categories.xlsx and candidates_round2_AS.xlsx (the old
two-workbook sources) are no longer read -- labelled.csv has superseded both.

What it guarantees:
  - schedule_related / job_control_related / not_related are clean 0/1 ints. A blank in any
    of the three means "not coded" and is treated as 0 -- these three columns are coded
    independently (each dimension marked only when it applies), so an uncoded cell is a
    true negative for that dimension, not a gap to fail on.
  - not_related is the exact complement of the other two (asserted, after the blank-fill
    above -- by construction each row has exactly one of {schedule, job_control} = 1, or
    neither, in which case not_related must be the one that's 1).
  - No id collides, and no review text appears twice -- both would mean the same review was
    coded more than once, possibly inconsistently. Exact duplicates are dropped (first
    occurrence wins) so that every surviving row gets exactly one out-of-fold prediction in
    15.aggregate_multilabel_cv.py.
  - The few-shot examples embedded in code/prompts/multilabel_system_prompt.txt get
    fold = -1: they sit in the system prompt for EVERY prediction, so letting one land in
    a test fold would leak its gold answer. fold == -1 rows always train, never test.
  - fold in {0..4}, stratified on the 4-way state key (2*schedule + job_control), fixed seed.

Run from the repo root:
    python code/10.prepare_multilabel_data.py
    python code/10.prepare_multilabel_data.py --input path/to/labelled.csv
"""

import argparse
import sys
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SOURCE_CSV = "data/trainingfinal/labelled.csv"
# labelled.csv mixes encodings across its rounds of hand-coding (Excel exports, pasted
# text); latin1 maps every byte 1:1 so it never raises, unlike utf-8 or cp1252 on this file.
SOURCE_ENCODING = "latin1"

OUT_CSV = "data/trainingfinal/labelled_cv.csv"

N_FOLDS = 5
RANDOM_SEED = 42

LABEL_COLS = ("schedule_related", "job_control_related", "not_related")
REQUIRED_COLS = ("id", "type", "review_text") + LABEL_COLS

STATE_NAME = {0: "none(00)", 1: "jobctrl(01)", 2: "sched(10)", 3: "both(11)"}

# ids of the few-shot examples hardcoded in the system prompt; must never be a test row
FEWSHOT_IDS = {
    45502, 78714, 30487, 13051, 98957,      # schedule-only (10)
    55488, 31910, 77436, 87495, 97736,      # job-control-only (01)
    69473, 78486, 89252, 51994,             # both (11)
    51350, 9892, 21959, 55390, 70129,       # neither (00)
}


def norm(text) -> str:
    """Whitespace/case-normalized key for duplicate detection."""
    return " ".join(str(text).split()).lower()


def load_source(path, encoding) -> pd.DataFrame:
    """Read the gold CSV and validate it."""
    df = pd.read_csv(path, encoding=encoding)
    print(f"Loaded {len(df)} rows from {path}")

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{path}: missing required column(s) {missing_cols}")

    # Each of the three label columns is coded independently (marked only when it applies),
    # so a blank cell means "not marked" = 0, not "not yet coded". Fill before validating.
    n_blank = int(df[list(LABEL_COLS)].isna().sum().sum())
    if n_blank:
        print(f"  Filling {n_blank} blank cell(s) across {list(LABEL_COLS)} with 0")
        df[list(LABEL_COLS)] = df[list(LABEL_COLS)].fillna(0)

    for col in LABEL_COLS:
        df[col] = df[col].astype(int)
        bad = set(df[col].unique()) - {0, 1}
        if bad:
            raise ValueError(f"{path}: non-binary values {bad} in {col!r}")

    # `subset` tags where each row came from (hand-coding batch); carry it through as
    # `source` for parity with 12.evaluate_multilabel.py / 15.aggregate_multilabel_cv.py,
    # which already know how to break results down by it.
    if "subset" in df.columns and "source" not in df.columns:
        df = df.rename(columns={"subset": "source"})

    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the outer-5-fold multilabel dataset.")
    ap.add_argument("--input", default=SOURCE_CSV, help="Path to the hand-coded labelled.csv.")
    ap.add_argument("--encoding", default=SOURCE_ENCODING, help="Encoding to read --input with.")
    ap.add_argument("--out", default=OUT_CSV, help="Output CSV path.")
    args = ap.parse_args()

    df = load_source(args.input, args.encoding)

    # --- duplicate ids --------------------------------------------------
    dupe_ids = df["id"][df["id"].duplicated()]
    if len(dupe_ids):
        raise ValueError(
            f"{dupe_ids.nunique()} id(s) appear more than once in {args.input}: "
            f"{sorted(dupe_ids.unique())[:20]}\n"
            f"  The same review coded twice must be resolved by hand before splitting."
        )

    # --- integrity: not_related is the complement of the other two ----
    either = ((df["schedule_related"] == 1) | (df["job_control_related"] == 1)).astype(int)
    mismatch = df["not_related"] != (1 - either)
    if mismatch.any():
        bad_ids = df.loc[mismatch, "id"].tolist()
        raise ValueError(
            f"{int(mismatch.sum())} row(s) where not_related != 1-(schedule|job_control): "
            f"{bad_ids[:20]}"
        )
    print("Integrity check passed: not_related == 1 - (schedule | job_control)")

    # --- 4-way stratification key (overwrites labelled.csv's stale `strat`, if any) ---
    df["strat"] = 2 * df["schedule_related"] + df["job_control_related"]
    print("\nState distribution:")
    for k, n in df["strat"].value_counts().sort_index().items():
        print(f"  {STATE_NAME[k]:>12}: {n}")
    if "source" in df.columns:
        print("\nState distribution by source:")
        print(pd.crosstab(df["strat"].map(STATE_NAME), df["source"]).to_string())

    # --- drop duplicate review texts ----------------------------------
    # Under an outer-fold partition a duplicated text is trained on twice and scored twice,
    # so 15.aggregate_multilabel_cv.py's "every row predicted exactly once" check would be
    # measuring noise. Drop them instead, keeping the first occurrence.
    df["text_key"] = df["review_text"].map(norm)
    for key, grp in df.groupby("text_key"):
        if grp["strat"].nunique() > 1:
            print(f"  WARNING: duplicate text with CONFLICTING labels "
                  f"(ids={grp['id'].tolist()}, states={grp['strat'].tolist()}): {key[:80]!r}")
    n_dupes = len(df) - df["text_key"].nunique()
    if n_dupes:
        dropped = df[df.duplicated("text_key", keep="first")]
        msg = f"\nDropping {n_dupes} duplicate review text(s)"
        if "source" in df.columns:
            msg += f" ({dict(dropped['source'].value_counts())})"
        print(msg + "; first occurrence kept.")
        df = df.drop_duplicates("text_key", keep="first").reset_index(drop=True)
    else:
        print("\nNo duplicate review texts.")

    # --- pin the few-shot rows out of the fold assignment -------------
    missing = FEWSHOT_IDS - set(df["id"])
    if missing:
        raise ValueError(
            f"Few-shot ids not found in {args.input}: {sorted(missing)}\n"
            f"  They are quoted verbatim in the system prompt and must exist as train rows."
        )

    is_pinned = df["id"].isin(FEWSHOT_IDS)
    pool = df[~is_pinned].reset_index(drop=True)

    # --- stratified outer folds ---------------------------------------
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_of = pd.Series(-1, index=pool.index, dtype=int)
    for k, (_, test_idx) in enumerate(skf.split(pool, pool["strat"])):
        fold_of.iloc[test_idx] = k
    if not (fold_of >= 0).all():
        raise AssertionError("StratifiedKFold left rows unassigned")

    df["fold"] = -1
    df.loc[~is_pinned, "fold"] = fold_of.to_numpy()

    # --- report -------------------------------------------------------
    print(f"\nOuter {N_FOLDS}-fold partition (seed {RANDOM_SEED}): "
          f"{int((df['fold'] >= 0).sum())} evaluable rows, "
          f"{int((df['fold'] == -1).sum())} pinned few-shot rows (always train, never test)")
    print("\nfold x state:")
    print(pd.crosstab(df["fold"], df["strat"].map(STATE_NAME)).to_string())
    if "source" in df.columns:
        print("\nfold x source:")
        print(pd.crosstab(df["fold"], df["source"]).to_string())
    print("\nfold x type (pros/cons):")
    print(pd.crosstab(df["fold"], df["type"]).to_string())

    # `set` (old 75/25 flag) is stale/absent for the newly added rows and superseded by
    # `fold` for every consumer except 13.GPT_multilabel_benchmark.R, which reads
    # labelled.csv directly and is unaffected by this script.
    drop_cols = [c for c in ("text_key", "set") if c in df.columns]
    out = df.drop(columns=drop_cols)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}  ({len(out)} rows, {out.shape[1]} cols)")
    print("Note: no `set` column -- consumers select rows by `fold`. "
          "13.GPT_multilabel_benchmark.R still reads the frozen labelled.csv and is unaffected.")


if __name__ == "__main__":
    sys.exit(main())
