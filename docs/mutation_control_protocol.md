# Mutation-only control

This exploratory ablation follows the encounter analysis. It compares crossover
plus mutation with mutation alone for the accelerating retained lineage.

The 40 accelerating runs from tempo pilot v1 are reused. Their source hashes,
Python and NEAT versions are checked before reuse, and their result hashes are
recorded in the new manifest. The old source files and results are not modified.

The new runs use the same Circle/Spiral generators, two data draws, seeds 220–229,
128 training examples, 512 test examples, four islands of 30 genomes, initial
hidden-node count, mutation settings, starting host and accelerating schedule.
The three 120-round phases use visit durations 15, 10 and 5. Each run evaluates
43,320 genomes, including 360 lineage candidates, with 43 host changes.

The only operator change is `crossover=False`: the lineage child is cloned from
the retained waltzer and mutated. Its insertion into the visited population and
strict improvement-based acceptance remain unchanged. Ordinary NEAT reproduction
still uses its usual crossover and mutation. Removing crossover changes RNG
consumption, so paired evolution seeds do not imply identical realized mutation
draws or ordinary population trajectories.

Report paired differences separately for each task/data draw, with ten evolution
seeds per comparison. Positive differences favor crossover plus mutation. Measure
best-global and final-waltzer test accuracy and MSE, alongside training traces.
Genomes are selected only by training MSE; no test-based selection or retuning is
performed. This is not fresh confirmatory data and does not isolate the causal
value of travel or prove transfer of particular capabilities.

```bash
python mutation_control.py --smoke-test --output-dir generated/mutation_control_smoke
python mutation_control.py --workers 3 --output-dir generated/mutation_control_v1
```

The smoke mode checks two small-data runs and does not compare them with the
full-data references. Full results include 40 new per-run JSON traces,
`scores.csv` (40 reused plus 40 new runs), paired differences, summaries, training
curves and provenance. Interrupted runs can continue with `--resume` only when
the protocol and recorded sources/references are unchanged.
