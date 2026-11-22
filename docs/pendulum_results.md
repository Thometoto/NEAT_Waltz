# Pendulum-v1 pilot results

Accelerating Waltz achieved higher held-out return than classic NEAT on eight of ten paired seeds in bank 0 and all ten in bank 1. Against the reversed schedule it won five of ten comparisons in each bank. The evidence favors the full Waltz configuration over the single-population baseline on this task. The contribution of visit order remains unresolved.

## Primary comparison

For each run, the champion was selected by training return across all evaluated genomes. Its test score is the mean undiscounted return over 32 held-out starts. Higher values are better. Each table entry below is accelerating Waltz minus classic NEAT in return units, paired by evolution seed. All summaries use unrounded scores.

| Evolution seed | Bank 0 | Bank 1 |
| --- | ---: | ---: |
| 300 | +92.02 | +40.51 |
| 301 | -31.75 | +496.14 |
| 302 | +363.39 | +35.94 |
| 303 | +386.82 | +263.02 |
| 304 | +101.89 | +240.49 |
| 305 | +26.86 | +146.94 |
| 306 | +3.56 | +49.40 |
| 307 | -121.48 | +203.84 |
| 308 | +243.73 | +310.09 |
| 309 | +171.86 | +1.87 |

## Comparisons with accelerating Waltz

Cells give the median paired difference and the counts of wins/ties/losses. Positive differences favor accelerating Waltz. The two banks use the same ten evolution seed labels with different training and test starts.

| Comparator | Bank 0 | Bank 1 |
| --- | ---: | ---: |
| Classic | +96.96 (8/0/2) | +175.39 (10/0/0) |
| Independent | -34.05 (5/0/5) | +75.16 (7/0/3) |
| Stationary | +45.19 (7/0/3) | +6.73 (7/0/3) |
| Constant | +3.30 (5/0/5) | +8.73 (5/0/5) |
| Decelerating | +2.56 (5/0/5) | +7.10 (5/0/5) |
| Nonpersistent | +83.85 (7/0/3) | +31.61 (6/0/4) |

Four independent islands yielded mixed comparisons across banks. The stationary lineage lost seven of ten pairs in both banks, while the moving constant-tempo and reversed-tempo controls each split the pairs evenly in both banks. This leaves open whether travel helps under other initial conditions and how much of the benefit comes from using separate populations.

## Absolute held-out returns

Each cell is the median across ten runs of the within-run mean test return. The secondary output is the final waltzer, except for nonpersistent crossover, where it is the best training-scored genome in the candidate stream. Classic NEAT and independent islands have only a primary output.

| Method | Best genome, bank 0 | Best genome, bank 1 | Secondary, bank 0 | Secondary, bank 1 |
| --- | ---: | ---: | ---: | ---: |
| Classic | -1019.70 | -877.43 | — | — |
| Independent | -792.07 | -866.93 | — | — |
| Stationary | -977.39 | -854.53 | -961.36 | -867.22 |
| Constant | -973.15 | -835.91 | -1001.42 | -836.81 |
| Accelerating | -878.97 | -836.37 | -892.32 | -866.53 |
| Decelerating | -967.44 | -850.62 | -1018.72 | -871.26 |
| Nonpersistent | -959.18 | -837.01 | -1064.70 | -898.59 |

## Coverage and interpretation

All 140 planned method/bank/seed combinations completed in 2 h 20 min 24 s with four workers. Each run used 43,320 genome evaluations and 34,656,000 training steps, for 4,851,840,000 training steps overall. Checks against the raw JSON covered source hashes, CSV entries, per-episode test means, reset seeds, population sizes, cumulative evaluations, champion scores and visit calendars. The six island methods also had matching generation-zero island scores within each bank and seed.

Both training and test starts change between banks. The experiment supports a task-specific comparison under the recorded settings. Ten evolutionary runs per bank and four fixed training starts per bank limit conclusions about other starts, schedules and tasks. All effects reported here are descriptive.

The [study protocol](pendulum_protocol.md) specifies the comparisons. Per-run JSON, episode returns, scores and timing are in `generated/pendulum_pilot_v1/`.

The [published result files](../results/pendulum_pilot_v1/) include the protocol and numerical tables.
