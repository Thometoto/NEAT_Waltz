# Mutation-only control: results

All 40 new mutation-only runs completed in 150.7 seconds with three workers.
They were paired with the 40 accelerating crossover-plus-mutation runs from
tempo pilot v1, under the [ablation protocol](mutation_control_protocol.md).
No historical source or result was modified.

## Verification

The 12 software tests and two small-data smoke runs passed. All 40 full-data
controls used 43,320 genome evaluations and 360 lineage candidates, with 43
host changes. Recorded local parents were null in every mutation-only child.
Population sizes, acceptance counts, archive scores and visit schedules passed
checks. Initial generation observations matched each paired crossover run
exactly. Reference file hashes and the frozen source hashes were verified.

The new controls used 1,732,800 genome evaluations and 221,798,400 training
predictions in total. The comparison tables contain 80 rows: 40 reused and
40 new, not 80 newly run experiments.

## Paired test results

Positive differences favor crossover plus mutation. Each value is the median
of ten within-seed differences for the same task/data draw, not a difference
between two separate medians. Accuracy differences are percentage points.

| Task | Data draw | Best-global accuracy gain | Final-waltzer accuracy gain | Best-global MSE reduction | Final-waltzer MSE reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| Circle | 0 | -0.88 | -0.10 | -0.00045 | -0.00149 |
| Circle | 1 | 0.00 | +4.49 | -0.00005 | +0.02879 |
| Spiral | 0 | -2.05 | +0.68 | -0.01574 | +0.02195 |
| Spiral | 1 | +0.29 | +17.19 | -0.00844 | +0.06244 |

For final-waltzer accuracy, crossover won on 5/10, 7/10, 5/10 and 10/10 seeds,
respectively. For final-waltzer test MSE, the corresponding counts were 5/10,
8/10, 7/10 and 9/10. Lower test MSE need not imply higher thresholded accuracy.

The zero median best-global accuracy difference on Circle draw 1 is not ten
ties: five differences were positive and five were negative.

## Interpretation

The clearest favorable result is for the final waltzer on Spiral draw 1:
the median accuracy gain is 17.19 points, with positive differences on all ten
seeds. The final-lineage test MSE also improves in this cohort. The benefit is
not uniform across data draws: Circle draw 0 has nearly zero median accuracy
change and a slightly unfavorable MSE difference.

This ablation supports the narrower observation that crossover with local
champions can help the retained lineage in some tested conditions, relative
to mutating it alone. It does not establish a general improvement of the full
algorithm: best-global test accuracy is mixed, and the median best-global test
MSE difference favors mutation-only in all four cohorts.

The lineage can therefore benefit from partners without producing the best
solution found by the populations. A stronger lineage score alone cannot show
that travel, acceleration or a persistent lineage is necessary. In particular,
the previous stationary-lineage control remains relevant.

Ordinary population crossover remains active in both methods. Removing the
special lineage crossover changes RNG consumption, so paired seeds do not mean
identical subsequent mutation draws or island trajectories. The comparison
measures the operator ablation within this implementation, not an isolated
causal effect of one inherited skill. The same two data draws and ten seeds per
draw were reused; no significance tests, confidence intervals or fresh
confirmatory evaluation were added.

## Next discriminating control

A nonpersistent crossover control would compare the current retained lineage
with children created from local and remote champions without preserving a
traveling parent. Matching insertion frequency and evaluation budgets would
test whether lineage continuity adds value beyond ordinary cross-island parent
selection. This control has not been run here. These results do not justify
changing the schedule to chase favorable scores.

## Files and reproduction

Run `python mutation_control.py --workers 3 --output-dir <fresh-directory>` from
the project directory. Completed runs can be resumed with `--resume` only when
the recorded protocol and reference/source hashes match.

Local outputs are in `generated/mutation_control_v1/`: the manifest, 40 per-run
JSON traces, `scores.csv`, `paired_differences.csv`, `paired_summary.csv`,
`lineage_progress.png`, `training_progress.png` and `timing.json`. The generated
directory is ignored by git; outputs need to be packaged explicitly for sharing.

The [published result files](../results/mutation_control_v1/) include the protocol and numerical tables.
