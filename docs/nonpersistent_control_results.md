# Nonpersistent crossover control: results

All 40 new controls completed in 154.7 seconds with three workers, under the
[recorded protocol](nonpersistent_control_protocol.md). They were paired with
the 40 existing accelerating Waltz runs. Historical files remain unchanged.

## Verification

Fifteen software tests and two small-data smoke runs passed. All 40 full-data
controls used 43,320 genome evaluations, 5,544,960 training predictions and 360
injected children. The 14,400 recorded parent pairs were checked against the
champion scores from the preceding evaluated generation. Parents were refreshed
each round, and no external breeding lineage appeared in the generation traces.

Initial island champion scores, the initial stream archive and all 44 visit
boundaries matched the paired Waltz runs. Both methods had 43 host changes.
Population sizes, archive updates, source hashes and all 40 reference hashes
passed verification. Comparison tables contain 80 rows: 40 reused and 40 new.

## Paired test results

Each entry is the median of ten within-seed differences. Positive values favor
persistent Waltz. Accuracy gains are percentage points; MSE reductions are
control MSE minus Waltz MSE. These are not differences between separate medians.

| Task | Data draw | Best-global accuracy gain | Best-global MSE reduction | Stream accuracy gain | Stream MSE reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| Circle | 0 | +3.03 | +0.01523 | +3.22 | +0.01108 |
| Circle | 1 | -3.03 | -0.00968 | 0.00 | -0.00635 |
| Spiral | 0 | +0.88 | -0.00381 | -0.39 | -0.00101 |
| Spiral | 1 | +2.44 | +0.00570 | +4.88 | +0.03801 |

For best-global accuracy, Waltz won/tied/lost on 6/1/3, 3/0/7, 5/0/5 and 7/0/3
seeds, respectively. For stream accuracy, these counts were 7/0/3, 5/0/5,
4/1/5 and 8/0/2. A zero median difference on Circle draw 1 does not mean that
all paired outcomes tied.

The stream output is the final retained waltzer for Waltz and the best
training-scored injected child (or the initial island-0 champion) for the control.
The control archive is diagnostic only and never used as a parent. Its selection
pool therefore matches the Waltz output pool without adding protected breeding
memory. The primary output remains the best-global genome.

## Interpretation

Persistent Waltz has favorable best-global accuracy differences in three of
four cohorts. Circle draw 0 and Spiral draw 1 also have favorable best-global
MSE differences. Circle draw 1 favors fresh champions, and Spiral draw 0 has
mixed metric rankings. The latter cohort's median accuracy advantage coexists
with five wins and five losses and an unfavorable MSE difference.

These observations identify conditions in which the retained-parent scheme
does better than this specific simple alternative. They do not establish a
consistent benefit across data draws. The positive stream result on Spiral
draw 1 is compatible with the earlier crossover-versus-mutation ablation, but
the same exploratory data are reused; this is not independent confirmation.

The control changes both the first-parent selection rule and the presence of
an externally retained breeding lineage. Ordinary populations still preserve
ancestry through selection and reproduction. This is not a memory-free search,
and a score difference cannot isolate the causal value of a latent memory
variable or demonstrate retained capabilities from particular islands.

Taken together, the current controls show that local-champion crossover can
help the traveling lineage and that persistence can matter in some cohorts.
They do not demonstrate that acceleration is necessary or that Waltz generally
outperforms standard NEAT. The current 361-generation pilot does not include a
matched standard-NEAT baseline; historical results used a different experiment.

Before a paper-level claim, the useful next step is a synthesis of the current
controls and a prespecified, more demanding task with fresh instances and a
matched standard-NEAT baseline. These point estimates from ten seeds and two
draws per task should not be presented as a general superiority result. No
significance claims or outcome-driven parameter changes were added here.

## Files and reproduction

Run `python nonpersistent_control.py --workers 3 --output-dir <fresh-directory>`.
Compatible interrupted runs can continue with `--resume`.

Outputs are in `generated/nonpersistent_control_v1/`: a protocol manifest,
40 per-run JSON files, `scores.csv`, `paired_differences.csv`,
`paired_summary.csv`, `training_progress.png`, `stream_progress.png` and
`timing.json`. Generated files are ignored by git and must be packaged explicitly
when sharing the results.

The [published result files](../results/nonpersistent_control_v1/) include the protocol and numerical tables.
