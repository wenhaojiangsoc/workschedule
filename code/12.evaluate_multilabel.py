"""
Evaluate one outer fold of the two-dimension classifier.

Under the nested outer-5-fold CV, model `outer{K}` was trained by 11.finetune_multilabel.py
on every row with `fold != K` and has never seen `fold == K`. This script scores it on
exactly those held-out rows. Run it for K = 0..4 and 15.aggregate_multilabel_cv.py pools
the five prediction files into one out-of-fold estimate covering the whole gold set.

Two modes:

  (1) Fine-tuned model (default):
        ML_OUTER_FOLD=0 python code/12.evaluate_multilabel.py
        python code/12.evaluate_multilabel.py --fold 0
      Loads the fold's LoRA adapter, generates two-digit predictions on that fold, prints
      the multi-label report, and writes data/trainingfinal/val_preds_<slug>_outer<K>.csv.

  (2) Score an existing predictions file with the *identical* metric code on the
      *identical* rows (this is how a closed-model benchmark stays comparable):
        python code/12.evaluate_multilabel.py --fold 0 --preds_csv path/to/preds.csv --name gpt-5.4
      The preds CSV must have an `id` column plus either `sched_pred`+`ctrl_pred`
      columns or a single two-digit `pred` column.
"""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_multilabel import parse_two_digit, multilabel_report
from multilabel_prompt import MODEL_SLUG, MAX_SEQ_LEN, generate_two_digit

DATA_CSV  = "data/trainingfinal/labelled_cv.csv"
TARGET_F1 = 0.95


def adapter_dir(fold):
    return f"models/finetuned/{MODEL_SLUG}_multilabel/outer{fold}"


def resolve_fold(cli_fold):
    """--fold wins; otherwise ML_OUTER_FOLD (so Slurm arrays need no extra plumbing)."""
    if cli_fold is not None:
        return cli_fold
    if "ML_OUTER_FOLD" in os.environ:
        return int(os.environ["ML_OUTER_FOLD"])
    raise SystemExit(
        "No outer fold selected. Pass --fold K or set ML_OUTER_FOLD=K.\n"
        "  All five folds: sbatch code/SBATCH/12.evaluate_multilabel.sh  (--array=0-4)"
    )


def load_test_set(fold):
    df = pd.read_csv(DATA_CSV)
    if "fold" not in df.columns:
        raise SystemExit(
            f"{DATA_CSV} has no `fold` column. Rebuild it with 10.prepare_multilabel_data.py."
        )
    if fold not in set(df["fold"]):
        raise SystemExit(f"fold {fold} not present in {DATA_CSV} "
                         f"(folds: {sorted(set(df['fold']))})")
    if fold < 0:
        raise SystemExit("fold -1 holds the pinned few-shot rows; they are never a test set.")

    test = df[df["fold"] == fold].reset_index(drop=True)
    print(f"Outer fold {fold}: {len(test)} held-out rows  "
          f"(schedule pos={int(test['schedule_related'].sum())}, "
          f"job_control pos={int(test['job_control_related'].sum())})")
    return test


def score_from_preds_csv(fold, path, name):
    """Score a predictions file against one fold's gold labels."""
    test = load_test_set(fold)
    preds = pd.read_csv(path)
    if "id" not in preds.columns:
        raise ValueError("preds CSV must contain an 'id' column to join on gold labels")

    if {"sched_pred", "ctrl_pred"}.issubset(preds.columns):
        preds = preds[["id", "sched_pred", "ctrl_pred"]].copy()
        n_malformed = 0
    elif "pred" in preds.columns:
        parsed = preds["pred"].map(parse_two_digit)
        preds = preds.assign(
            sched_pred=[p[0] for p in parsed],
            ctrl_pred=[p[1] for p in parsed],
        )[["id", "sched_pred", "ctrl_pred"]]
        n_malformed = int(sum(not p[2] for p in parsed))
    else:
        raise ValueError("preds CSV needs 'sched_pred'+'ctrl_pred' or a 'pred' column")

    # A preds file may legitimately cover more than one fold; keep only this fold's rows.
    preds = preds[preds["id"].isin(set(test["id"]))]
    merged = test.merge(preds, on="id", how="left", validate="one_to_one")
    missing = merged["sched_pred"].isna().sum()
    if missing:
        raise ValueError(f"{missing} of fold {fold}'s test rows have no prediction in {path}")

    multilabel_report(
        merged["schedule_related"], merged["job_control_related"],
        merged["sched_pred"].astype(int), merged["ctrl_pred"].astype(int),
        name=f"{name} (outer fold {fold})", target_f1=TARGET_F1, n_malformed=n_malformed,
    )


def score_finetuned_model(fold):
    os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
    os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from unsloth import FastModel

    test = load_test_set(fold)

    a_dir = adapter_dir(fold)
    if not os.path.exists(a_dir):
        raise SystemExit(
            f"No adapter at {a_dir}. Train it first:\n"
            f"  ML_OUTER_FOLD={fold} python code/11.finetune_multilabel.py"
        )

    print(f"Loading model + adapter from {a_dir} ...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=a_dir,
        max_seq_length=MAX_SEQ_LEN,  # match training; see multilabel_prompt.py
        load_in_4bit=True,
        load_in_16bit=False,
        full_finetuning=False,
        device_map="balanced",
    )
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer

    print("Running predictions...")
    raw, sp, cp, malformed = [], [], [], 0
    for i, (_, row) in enumerate(test.iterrows()):
        text = generate_two_digit(model, tokenizer, row["review_text"], max_new_tokens=8)
        s, c, ok = parse_two_digit(text)
        malformed += (not ok)
        raw.append(text); sp.append(s); cp.append(c)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(test)} done")

    keep = [c for c in ("id", "review_text", "type", "source",
                        "schedule_related", "job_control_related") if c in test.columns]
    out_df = test[keep].copy()
    out_df["fold"] = fold
    out_df["sched_pred"] = sp
    out_df["ctrl_pred"] = cp
    out_df["raw_output"] = raw
    out_path = f"data/trainingfinal/val_preds_{MODEL_SLUG}_outer{fold}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

    multilabel_report(
        test["schedule_related"], test["job_control_related"], sp, cp,
        name=f"{MODEL_SLUG} finetuned (outer fold {fold})",
        target_f1=TARGET_F1, n_malformed=malformed,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None,
                    help="Outer fold to evaluate (defaults to $ML_OUTER_FOLD).")
    ap.add_argument("--preds_csv", default=None,
                    help="Score an existing predictions CSV instead of loading the model.")
    ap.add_argument("--name", default=None, help="Label for the report header.")
    args = ap.parse_args()

    fold = resolve_fold(args.fold)
    if args.preds_csv:
        score_from_preds_csv(fold, args.preds_csv,
                             args.name or os.path.basename(args.preds_csv))
    else:
        score_finetuned_model(fold)


if __name__ == "__main__":
    main()
