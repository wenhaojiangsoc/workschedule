# corporate-control

Measuring work-schedule and job-control experience from employee reviews, and linking it
to firm ownership and governance.

The pipeline has three parts. Scripts are numbered in run order; a gap in the numbering
means a step that lives in another part.

```
reviews ──► 03–09   binary screen: is this excerpt about scheduling at all
        └─► 10–15   stage 1 gate: schedule_related / job_control_related  (2 digits)
        └─► 20–25   stage 2 dimensions: which of the 27 dimensions apply
            30      learning curves: does more hand-coding still pay
            40–41   active sampling: choose the next coding batch with the trained model
        └─► linkage/  Glassdoor employer ──► Orbis firm ──► ownership panel
```

## Where the numbers stand

Five-fold nested cross-validation over 1,578 hand-coded excerpts.

| Model | Metric | Score |
|---|---|---|
| Stage 1 gate | macro F1 | 0.944 (sd 0.003 across folds) |
| 20 schedule dimensions | pooled micro F1 | 0.929, macro 0.888, exact-match 0.874 |
| 7 job-control dimensions | pooled micro F1 | 0.948, macro 0.912, exact-match 0.908 |
| Gate → dimensions, end to end | micro F1 | 0.903, macro 0.861 |

The end-to-end row is the honest production number: the gate routes, so its errors
compound into stage 2. The two rows above it are scored with the true branch given.

For scale, two human coders working independently on the same excerpts agree at macro F1
0.771 pooled, 0.885 for the strongest pair (`code/25.report.py` prints this alongside the
model numbers).

## The model

Qwen3.5-35B-A3B, 4-bit, LoRA on attention projections only (`q,k,v,o`). Including the
MoE expert layers attaches adapters to all 256 experts and exhausts memory, which is what
killed the earlier run. Trainable parameters are 1.7M to 13.8M depending on rank, against
35B total.

Optuna searches rank {8,16,32,64}, alpha {r, 2r}, and learning rate 5e-5 to 3e-4 inside
each outer fold, so no hyperparameter information crosses into the held-out fold. Epochs
cap at 5 or 6 but early stopping ends training at 3.0 on average.

## Running it

One submission trains everything:

```
sbatch code/SBATCH/run_everything.sh
```

15 array tasks: 0–4 the gate, 5–9 the schedule dimensions, 10–14 job control, plus the
three learning curves. Each task resumes from its last checkpoint, keeps its own Optuna
study, and resubmits itself until it writes a `DONE` marker, so timeouts and preemption
need no attention.

Check progress at any time:

```
python code/25.report.py
```

### Cluster notes

Sized for Duke DCC. Two things will bite on a different cluster:

- `run_everything.sh` sizes the training batch from `nvidia-smi` memory and the branch,
  holding the *effective* batch at 8. Stage-2 prompts run ~2,000 tokens and the logits
  tensor is upcast to fp32, which is what exhausts a 32GB card at batch 8.
- `UNSLOTH_COMPILE_LOCATION` must be per-task. Unsloth writes generated source to
  `$PWD/unsloth_compiled_cache` and the training scripts clear it before each model load;
  concurrent tasks sharing one working directory race and the loser dies with
  `OSError: could not get source code`.

## Layout

```
code/
  00–02   descriptives (R, notebook, quarto)
  03–05   binary scheduling screen
  06–09   binary fine-tune, evaluation, corpus labelling
  10–15   stage 1: prepare folds, fine-tune, evaluate, aggregate, GPT benchmark
  20–23   stage 2: prepare branches, fine-tune, evaluate pooled and end-to-end
  25      report: model numbers, human ceiling, per-dimension labelling verdict
  30      learning curves for all three models
  40–41   score a mined candidate pool, select the next coding batch
  multilabel_prompt.py, hier_prompt.py   prompt format and decoding, shared by
                                          training and inference so they cannot drift
  metrics_multilabel.py                   two-digit parsing
  prompts/                                system prompts (stage 1, and one per branch)
  SBATCH/                                 Slurm launchers
  linkage/                                Glassdoor ↔ Orbis crosswalk
```

See `docs/METHODS.md` for the measurement design and how to read the evaluations.

Data, model adapters and figures are not in the repository. Scripts expect
`data/trainingfinal/` beside `code/`.

## Known gaps

- `SBATCH/16.ensemble_label.sh` calls `code/16.ensemble_label.py`, which is not in the
  repository. The five-model majority vote it describes has never been run. Given the
  folds agree to within 0.003 macro F1, test one model against the ensemble before paying
  3.2× the inference cost.
- `linkage/` was written in a separate work session, not alongside the classifier. It is
  included so the pipeline is complete end to end. See `TECHNICAL_REPORT.md` and
  `JOIN_GUIDE.md` in the linkage output directory, which are not shipped here.
- Nine of the 27 dimensions have fewer than 20 positives and their F1 is not yet
  trustworthy. `code/30.saturation.py` and `code/25.report.py` identify which.
