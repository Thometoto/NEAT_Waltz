# Tempo pilot v1: results

All 200 planned runs completed on 2026-11-18. The observed wall time was
715.9 seconds (11.9 minutes) with three worker processes. This is an exploratory
pilot under the [prespecified protocol](tempo_pilot_protocol.md), not a
confirmatory comparison against standard NEAT or migration.

## Design and verification

Circle and Spiral each used two independent training/test draws, ten evolution
seeds per draw and five methods. Each run used four populations of 30 genomes,
361 evaluated generations and, where applicable, one waltzer. All runs used
43,320 training genome evaluations and 5,544,960 training predictions. The full
lot therefore used 8,664,000 genome evaluations and 1,108,992,000 predictions.

The accelerating and decelerating schedules used equal 120-generation phases
and 43 host changes. Constant tempo 10 used 35 host changes. Every Waltz variant
evaluated 360 candidate children within the common budget. Independent islands
used ordinary reproduction instead.

All saved runs passed checks of job identity, source hashes, population sizes,
evaluation counts, visit acceptance totals, archive scores and monotone lineage
training fitness. The ten repeated benchmark cases reproduced the same scores
and observations in the pilot, excluding elapsed time. Twelve software tests
and the ten-run small-data smoke workflow also passed.

## Paired test results

Each cell below is the median of ten within-seed accuracy differences, in
percentage points. Positive values favor acceleration. These are **not**
differences between the methods' separate median accuracies.

Data draw 0 uses train/test seeds 101/102; draw 1 uses 103/104. Draws are reported
separately and are not pooled as independent benchmark problems.

| Task | Data draw | Best genome: vs constant | Best genome: vs decelerating | Final waltzer: vs constant | Final waltzer: vs decelerating |
| --- | --- | ---: | ---: | ---: | ---: |
| Circle | 0 | -0.29 | +2.54 | +1.27 | +0.78 |
| Circle | 1 | -3.52 | -6.15 | -2.15 | -3.81 |
| Spiral | 0 | -1.07 | -1.17 | -1.76 | -2.44 |
| Spiral | 1 | +0.29 | +1.37 | +6.35 | +2.93 |

For the best genome, acceleration beat deceleration on 7/10 evolution seeds on
Circle draw 0, but only 2/10 on draw 1. On Spiral the corresponding counts were
3/10 (one tie) and 7/10. For the final waltzer on Spiral draw 1, acceleration beat
constant tempo on 9/10 seeds; on draw 0 it won on 3/10.

## Interpretation

Acceleration did not produce a consistent advantage across these four
task/draw combinations. Its effect changed direction across data draws on both
tasks. The positive final-waltzer result on Spiral draw 1 is an observation to
investigate, not evidence that acceleration is generally preferable.

The observations also show why the retained waltzer and best overall genome
must be distinguished. Ordinary island evolution can find a stronger model
than the final lineage. Training MSE and test accuracy can rank methods
differently; genome selection used training MSE throughout.

This pilot neither validates nor rules out the proposed accumulation mechanism.
It tests one schedule, population arrangement, starting host and evaluation
budget on two small tasks. There are only two data draws per task. No uncertainty
intervals, significance claims or outcome-based parameter tuning are presented.
The constant-tempo comparison also changes travel count, while the reversed
schedule matches it. Broader claims require more demanding tasks, fresh
confirmation data and migration/nonpersistent-lineage controls.

The next useful analysis is to inspect the saved encounter traces: which visits
produce accepted children, how the gains change over time and how the lineage
compares with the ordinary populations. Gains alone cannot establish transfer
because the children undergo crossover and mutation together. These results do
not justify adding complexity or searching schedules until one wins.

## Files

The complete local outputs are in `generated/tempo_pilot_v1/`:

- `protocol.json`: frozen settings, job list and source hashes.
- `scores.csv`: all 200 per-run scores and encounter summaries.
- `paired_summary.csv`: paired differences for all controls and both outputs.
- `training_progress.png`: median best-global training MSE across seeds.
- `lineage_progress.png`: median retained-waltzer training MSE across seeds.
- Per-run JSON files: candidate, generation and visit observations.
- `timing.json`: wall time and worker count.

The generated directory is ignored by git. The results remain available locally;
publishing this report requires explicitly packaging the associated outputs.

The [published result files](../results/tempo_pilot_v1/) include the protocol and numerical tables.
