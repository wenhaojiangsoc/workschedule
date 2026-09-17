"""
Single source of truth for the stage-2 (dimension-level) prompt format and decoding.

Mirrors multilabel_prompt.py, which does the same job for the stage-1 gate. Imported by
21.finetune_hierarchical.py and 22.evaluate_hierarchical.py so training and inference
cannot drift.

Two branches:
  sc : 20 schedule dimensions   (positions 1-10 cons, 11-20 pros)
  jc : 7  job-control dimensions (positions 1-5 cons, 6-7 pros)

Torch-free at import time. Runnable on a login node to check the token budget:
    python code/hier_prompt.py sc data/trainingfinal/hier_sc_cv.csv
"""

import os

MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"
MODEL_SLUG = MODEL_NAME.split("/")[-1]

# The sc prompt is the long one (~1.5k tokens) and reviews run to ~120 tokens.
MAX_SEQ_LEN = 3072
ANSWER_MARGIN = 8

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

BRANCH = {
    "sc": {"dims": SC_CONS + SC_PROS, "n_cons": len(SC_CONS), "prompt": "hier_sc_system_prompt.txt"},
    "jc": {"dims": JC_CONS + JC_PROS, "n_cons": len(JC_CONS), "prompt": "hier_jc_system_prompt.txt"},
}

_HERE = os.path.dirname(os.path.abspath(__file__))


def dims_for(branch):
    return BRANCH[branch]["dims"]


def n_digits(branch):
    return len(BRANCH[branch]["dims"])


def system_prompt(branch):
    with open(os.path.join(_HERE, "prompts", BRANCH[branch]["prompt"]), encoding="utf-8") as f:
        return f.read().strip()


def format_prompt(branch, doc, polarity):
    """Inference prompt. `polarity` is 'c' or 'p' and is stated explicitly: the model
    needs it to know which half of the digit string is even eligible."""
    tag = "CONS" if polarity == "c" else "PROS"
    return (
        f"<|im_start|>system\n{system_prompt(branch)}<|im_end|>\n"
        f"<|im_start|>user\n[{tag}] {doc}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def format_answer(target):
    return f"{target}<|im_end|>"


def format_example(branch, doc, polarity, target):
    return format_prompt(branch, doc, polarity) + format_answer(target)


def parse_digits(text, branch):
    """Return (list_of_ints, ok). ok is False if the model did not emit exactly n digits."""
    n = n_digits(branch)
    t = text.strip()
    if "</think>" in t:
        t = t.split("</think>", 1)[1].strip()
    keep = "".join(ch for ch in t if ch in "01")
    if len(keep) != n:
        return [0] * n, False
    return [int(c) for c in keep], True


def im_end_id(tokenizer):
    tid = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if tid is None or tid < 0:
        return tokenizer.eos_token_id
    return tid


def generate_digits(model, tokenizer, branch, doc, polarity, max_new_tokens=None):
    import torch
    if max_new_tokens is None:
        max_new_tokens = n_digits(branch) + 6
    inputs = tokenizer(format_prompt(branch, doc, polarity), return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             eos_token_id=im_end_id(tokenizer),
                             pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    del inputs, out
    return text.strip()


def preflight_token_budget(branch, df, max_seq_len=None):
    """Fail in seconds if max_seq_len would truncate the answer digits."""
    from transformers import AutoTokenizer
    import numpy as np
    max_seq_len = MAX_SEQ_LEN if max_seq_len is None else max_seq_len
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    n_sys = len(tok(system_prompt(branch), add_special_tokens=False)["input_ids"])
    lens = np.asarray([
        len(tok(format_example(branch, r["review_text"], r["type"], str(r["target"])),
                add_special_tokens=False)["input_ids"])
        for _, r in df.iterrows()
    ])
    need = int(lens.max()) + ANSWER_MARGIN
    print("\n" + "=" * 60)
    print(f"PREFLIGHT: token budget (branch {branch})")
    print(f"  system prompt              : {n_sys}")
    print(f"  full example min/med/max   : {lens.min()} / {int(np.median(lens))} / {lens.max()}")
    print(f"  required (max + {ANSWER_MARGIN} margin) : {need}")
    print(f"  MAX_SEQ_LEN                : {max_seq_len}")
    print("=" * 60)
    if need > max_seq_len:
        raise SystemExit(
            f"FATAL: MAX_SEQ_LEN={max_seq_len} truncates training examples; need {need}. "
            f"Set MAX_SEQ_LEN >= {((need + 63) // 64) * 64} in code/hier_prompt.py.")
    w = df.iloc[int(np.argmax(lens))]
    ids = tok(format_example(branch, w["review_text"], w["type"], str(w["target"])),
              add_special_tokens=False)["input_ids"][:max_seq_len]
    tail = tok.decode(ids[-(n_digits(branch) + 4):])
    if str(w["target"]) not in tail:
        raise SystemExit(f"FATAL: answer missing from the tail after truncation; tail={tail!r}")
    print(f"  answer survives truncation : OK (tail = {tail!r})\n")


if __name__ == "__main__":
    import sys
    import pandas as pd
    br = sys.argv[1] if len(sys.argv) > 1 else "sc"
    csv = sys.argv[2] if len(sys.argv) > 2 else f"data/trainingfinal/hier_{br}_cv.csv"
    df = pd.read_csv(csv, dtype={"target": str})
    print(f"branch {br}: {len(df)} rows, {n_digits(br)} digits")
    preflight_token_budget(br, df)
    print("Preflight OK.")
