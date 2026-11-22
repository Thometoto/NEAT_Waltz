# Tempo pilot v1

This exploratory pilot tests whether the order of visit durations changes the
progress of a single retained lineage. It is not a confirmatory paper experiment.
The protocol is specified before examining new pilot outcomes.

## Fixed design

- Tasks: Circle and Spiral, using their existing generators without added noise.
- Each task uses training/test seed pairs (101, 102) and (103, 104).
- Each instance contains 128 balanced training examples and 512 independent test
  examples. All methods and evolution seeds share the same data for that instance.
- Evolution seeds: 220 through 229 inclusive, crossed with both data pairs and
  both tasks. Five methods produce 200 runs in total.
- Four islands of 30 genomes; one external waltzer, initially on island 0; one
  crossover-plus-mutation child per candidate round, using the local champion.
- Initial hidden nodes: 4. Weight/bias mutation standard deviation: 0.1.
  Remaining settings come from the shared configuration. The manifest records
  source hashes, settings and dependency versions.
- 361 evaluated generations including initialization: 43,320 training genome
  evaluations and 5,544,960 training predictions per run. A Waltz run evaluates
  exactly 360 lineage candidates within this budget. Independent islands use
  those places for ordinary offspring instead.

| Method | Candidate evaluation rounds 1–120 | 121–240 | 241–360 | Host changes |
| --- | --- | --- | --- | --- |
| Accelerating | 15 generations per visit | 10 | 5 | 43 |
| Constant | 10 | 10 | 10 | 35 |
| Decelerating | 5 | 10 | 15 | 43 |
| Stationary | Same visit boundaries as accelerating | Same | Same | 0 |
| Independent | No waltzer | No waltzer | No waltzer | 0 |

Each scheduled phase contains complete four-island tours. Acceleration and
deceleration therefore have identical phase durations, visit counts and total
travel counts; only their ordering differs. Constant tempo 10 does not match
travel count. Its comparison measures the entire schedule, not ordering alone.
Host changes count transitions between evaluated visits; no unused departure
after the last evaluation is counted. Visits end at rounds 120, 240 and 360.

## Outputs and interpretation

Primary contrasts: accelerating versus constant, and accelerating versus
decelerating. Stationary lineages and independent islands provide additional
controls. The best genome seen is selected by training MSE. The final waltzer is
reported separately. Test data never select genomes, tempo or partners.

Record all candidate and generation events, visit summaries, acceptance rate,
accepted fitness gains, travel count and training time. Compare training curves
against evaluations. Summarize paired test-accuracy differences across evolution
seeds separately for each task and data instance. Report wins/ties/losses and
median differences; do not treat the 20 task-instance runs as 20 independent
datasets. No significance claims or tuning on test outcomes are planned here.

These observations do not establish that improvements are caused by transferred
capabilities: crossover and mutation occur together. The pilot has no standard
NEAT, migration or nonpersistent-lineage control. Those remain necessary for
broader claims. Circle and Spiral are small preliminary tasks; a later study
needs more demanding problems and fresh confirmation seeds/instances.

## Execution

From the project directory, with the virtual environment activated:

```bash
python -m unittest test_waltz_observations -v
python pilot_tempo.py --smoke-test --workers 3 --output-dir generated/tempo_pilot_smoke
python pilot_tempo.py --benchmark --workers 3 --output-dir generated/tempo_pilot_benchmark
python pilot_tempo.py --workers 3 --output-dir generated/tempo_pilot_v1
```

The smoke workflow uses small datasets and ten runs. The benchmark uses the full
budget and datasets on the first seed/instance: ten runs whose timing estimates
the full lot. Benchmark outcomes do not change the protocol. Benchmark runs are
kept separate and are not added to the 200 pilot runs.

Each completed run is saved immediately. An interrupted lot can continue with
`--resume` and the same output directory only if the protocol and source hashes
match exactly. Existing outputs are not silently overwritten. The final output
contains `protocol.json`, per-run JSON traces, `scores.csv`,
`paired_summary.csv`, `training_progress.png`, `lineage_progress.png` and wall-time information.
