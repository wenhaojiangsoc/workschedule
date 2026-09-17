# Changes against the original repository

Two kinds of change: revisions to existing scripts, and new scripts for steps that were
not in the original. Nothing in `00`–`09`, `13`, `15` was touched.

## Revised

### `code/11.finetune_multilabel.py`

**Real batching.** Was `per_device_train_batch_size=1` with `gradient_accumulation_steps=8`;
now both are environment-tunable and default to a real batch with the *effective* batch
held at 8. That recipe was tuned for a 47GB card. Measured on an H200: 18.1 s/step before,
3.35 s/step after, a 5.4× speedup with identical optimization.

**Objective changed from schedule F1 to macro F1.** Schedule F1 saturates near 0.95 across
every configuration, so it could not separate trials; the original code already
acknowledged this by using macro F1 as the tie-break. Macro F1 is now primary and
job-control F1 the tie-break, since job control is the binding dimension.

**Resume instead of restart.** `trainer.train(resume_from_checkpoint=...)` when a
checkpoint exists. Without it a preemption or 24h timeout discarded the whole fold.

**Checkpoint interval 100 → 50 steps**, matching the eval interval so it costs no extra
eval passes. Preemptible partitions give no warning.

**`UNSLOTH_COMPILE_LOCATION` honoured** when clearing the compile cache. The original
hardcoded `$PWD/unsloth_compiled_cache`; with concurrent array tasks in one working
directory this races and the loser dies with `OSError: could not get source code`. This
was observed killing 191 launches.

**`DONE` marker** written on completion, so a launcher can tell a finished task from a
killed one.

`RECIPE_VERSION` bumped `v3` → `v4` so none of this silently resumes an older study.

### `code/SBATCH/11.finetune_multilabel.sh`

Rewritten for a different Slurm account. The original requested 5 GPUs for 7 days on
partition `h200ea`; the account used here is capped at 2 concurrent GPUs and 24h, so the
five folds run as an array of single-GPU tasks rather than one node-wide job. Superseded
by `run_everything.sh`, kept for provenance.

## New

| Script | Purpose |
|---|---|
| `20.prepare_hierarchical_data.py` | Build the two stage-2 branch datasets, inheriting stage-1 folds so a review held out by the gate is held out downstream |
| `21.finetune_hierarchical.py` | Stage-2 fine-tune, N-digit multi-label target, one model per branch |
| `22.evaluate_hierarchical.py` | Pooled out-of-fold scoring, so per-dimension F1 rests on full support rather than one fold |
| `23.evaluate_pipeline.py` | End-to-end: the gate routes into stage 2, the honest production number |
| `25.report.py` | Model numbers, human ceiling from the double-coded subsets, per-dimension labelling verdict |
| `30.saturation.py` | Learning curves: F1 against training-set size, fixed held-out fold, nested stratified subsets, two seeds |
| `40.score_pool.py` | Batched scoring of a mined candidate pool with the trained models |
| `41.select_round4.py` | Select the next coding batch: confirmations, contradictions and random anchors |
| `hier_prompt.py` | Stage-2 prompt format and decoding, shared by training and inference |
| `prompts/hier_{sc,jc}_system_prompt.txt` | Stage-2 system prompts carrying the codebook |
| `SBATCH/run_everything.sh` | Single launcher for all 15 training tasks, self-resuming |

## Codebook and labels

The training labels changed on 2026-09-04, before any of this was run. 205 cells across
174 excerpts were corrected, and two dimensions were added: *uneven or unfair allocation
of work* and *inconsistent or cyclical workload over time*, taking the codebook from 25
dimensions to 27. `code/coding_changes_2026-09-04*.R` applies those changes and logs every
cell it touches.

The effect was large and concentrated in the weak dimension: job-control F1 went from
0.595 on the pre-correction labels to 0.878 on the corrected ones in an otherwise
identical smoke test. Any run against a `labelled_cv.csv` with 578 rather than 607
job-control positives is using the stale labels.

## Bugs found and fixed

- **`30.saturation.py`**: `pool.index[...].to_numpy()` returns a read-only array, and
  `rng.shuffle` refuses it. Would have crashed all three learning-curve tasks at startup.
- **Batch-size override**: the launcher exported a fixed batch after the submit-time
  values, silently overriding them and exhausting memory on smaller cards.
- **Resume fingerprint**: `train_bs` and `grad_accum` were in the Optuna resume
  fingerprint, so a task resuming on a differently sized card was refused. Removed, since
  the effective batch is what matters.
