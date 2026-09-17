# Measurement design

How an employee review becomes a firm-year measure of schedule control, and why the
pipeline is shaped this way.

## The problem

A review's "cons" box might read *"mandatory overtime, no notice, and they cut your hours
in January"*. That is three distinct scheduling complaints. Counting keywords cannot
separate them, and a flat 27-way classifier trained on 1,578 examples cannot either,
because most of the 27 dimensions appear in fewer than 40 of them.

So the task is split in two.

## Stage 1: the gate

Every excerpt gets two binary labels, emitted as two digits:

- `schedule_related` — timing, duration, predictability, flexibility of work hours, plus
  work location
- `job_control_related` — intensity and organisation of work *during* hours, and autonomy
  that is not about the schedule

These are independent, not exclusive. An excerpt can be both, one, or neither. Of a random
corpus sample, 22.2% are schedule-related and 10.8% job-control.

The gate is where the hard conceptual boundary sits. "Flexible" with no scheduling cue is
job control; "flexible hours" is schedule. "Understaffed" is job control unless the review
also mentions hours or shifts.

## Stage 2: dimensions within a branch

An excerpt the gate accepts goes to a branch model that emits one digit per dimension:

- **sc**, 20 digits — 10 schedule complaints, then 10 schedule compliments
- **jc**, 7 digits — 5 job-control complaints, then 2 compliments

Polarity is given to the model as a `[CONS]` or `[PROS]` tag but is **not** hard-masked.
The codebook allows cross-polarity coding: a complaint sitting in the pros box is coded
for what it says. This is rare, 1% of schedule rows, and masking it measured slightly
worse (micro F1 0.9287 unmasked against 0.9274 masked), so the prior is soft.

## Why folds are inherited

Stage 2 reuses the exact fold assignment from stage 1 rather than drawing its own. A review
held out by the gate is held out by the dimension models too. Without this, the end-to-end
evaluation would route a review through a stage-2 model that had trained on it.

## Three evaluations, and which to quote

| Script | What it measures | When to use it |
|---|---|---|
| `21` (during training) | inner-CV F1 within each fold's training pool | hyperparameter selection only |
| `22.evaluate_hierarchical.py` | pooled out-of-fold, gold branch given | per-dimension quality, full support |
| `23.evaluate_pipeline.py` | gate routes into stage 2 | **the production number** |

The gap matters. Gold-gated micro F1 is 0.929 and 0.948 for the two branches; end to end it
is 0.903, because gate errors compound. Quote the end-to-end figure for anything involving
the full corpus.

## Saturation

`30.saturation.py` holds one fold out and trains on nested, stratified subsets of the rest,
two seeds per point. Hyperparameters are fixed, because a learning curve must vary one
thing at a time.

Findings as of the 1,578-excerpt training set:

- **The gate is saturated.** Schedule F1 sits at 0.95 from the first 100 training rows and
  never moves. Only job control improves, 0.846 to 0.924, and flattens past 600.
- **Job-control dimensions are saturated**, gaining about half an F1 point per 100 rows.
- **Schedule dimensions still pay**, about 0.7 points per 100 rows past n=300. Doubling
  that branch would plausibly buy 2 to 4 points.

## Active sampling

`40` and `41` choose the next coding batch using the trained model rather than a regex.
Three kinds of row go in deliberately:

- **confirmed** — the model predicts the dimension; fills the support gap. Allocated as
  `gap / F1`, so a dimension the model is shakier on gets more candidates to survive its
  own false positives.
- **contradiction** — keyword search said yes, model said no. These sit on the boundary the
  model currently gets wrong and teach more per coded row.
- **anchor** — random corpus excerpts, never keyword-matched.

The anchors are not optional. A batch built only from model agreements confirms what the
model already believes and bakes its blind spots into the next training round.

## Interpreting the numbers

Two humans coding the same excerpts independently agree at macro F1 0.771 pooled and 0.885
for the strongest pair. The model scores 0.944 against adjudicated gold. These are not
like-for-like, since the model is scored against a consensus the humans were used to build,
but they bound what agreement on this task looks like.

Nine of 27 dimensions have fewer than 20 positives. Their F1 is not trustworthy at any
training size and the reported values swing by up to 0.17 across folds. `25.report.py`
flags these explicitly rather than reporting them alongside the solid ones.
