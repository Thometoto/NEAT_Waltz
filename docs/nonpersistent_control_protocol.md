# Nonpersistent crossover control

This exploratory control compares persistent accelerating Waltz with fresh
cross-island champion mating. The 40 existing accelerating runs are reused;
40 new runs use the same data draws, seeds, configuration, populations and budget.

Every generation inserts one crossover-plus-mutation child into the scheduled
host. The host follows Waltz's accelerating 15/10/5 calendar, with three equal
120-round phases, 360 evaluated children and 43 host changes. All runs evaluate
43,320 genomes (5,544,960 training predictions).

The local parent is the host's current champion. The other parent is the current
champion of `(host - 1) % 4`. Both are snapshotted from the evaluated populations
before any island reproduces. They are selected afresh every generation. Children
remain in ordinary populations and can reproduce normally; no external parent
is protected, carried between islands or replaced through a lineage acceptance
gate. This removes external breeding persistence, not all genetic ancestry or
selection inside the populations.

The best-global genome is the primary output. A separate diagnostic archive
selects the best training-scored genome among the initial island-0 champion and
all injected children. This matches the selection pool used by the final Waltz
lineage. The diagnostic archive is never a breeding parent. It is not called a
final waltzer in this control.

Paired differences are reported separately for each task/data draw, across ten
evolution seeds. Positive differences favor persistent Waltz. Scores are selected
only on training MSE; accuracy and MSE on test data are measured afterward.
Reference source, dependency and result hashes are checked before reuse.

This comparison changes the first-parent rule as well as removing the protected
lineage: a fresh neighboring champion replaces a retained, possibly older genome.
It tests whether the full persistent-parent scheme beats this explicit simple
alternative. It does not isolate an abstract memory variable, test every possible
donor policy, or establish capability transfer. These reused data are exploratory.

```bash
python -m unittest test_nonpersistent_control test_waltz_observations -v
python nonpersistent_control.py --smoke-test --output-dir generated/nonpersistent_control_smoke
python nonpersistent_control.py --workers 3 --output-dir generated/nonpersistent_control_v1
```

The two-run smoke workflow uses small datasets and is not compared to the full
pilot. Full outputs include 40 new per-run traces, 40 paired differences, summaries,
80 score rows (reused plus new), training plots and provenance. A compatible
interrupted lot can continue with `--resume`; existing nonempty output directories
are otherwise refused.
