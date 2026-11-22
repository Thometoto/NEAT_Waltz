# Pendulum confirmation results

Accelerating Waltz had a positive median advantage over classic NEAT in six of eight new banks of starting states. Against four independent islands it had a positive median advantage in three banks. Both outcomes fell below the prespecified descriptive threshold of seven favorable banks.

## Paired held-out returns

Each run contributed the mean return of its training-selected champion on 32 held-out starts. Within each bank, the five paired differences use matching evolution seeds. Each cell gives the median difference in return units (positive favors accelerating Waltz), followed by wins/ties/losses for Waltz. The eight bank medians receive equal weight.

| Bank | Classic NEAT | Four independent islands | Stationary waltzer |
| --- | ---: | ---: | ---: |
| 2 | +162.65 (5/0/0) | -27.25 (2/0/3) | -44.40 (2/0/3) |
| 3 | -44.92 (2/0/3) | +3.17 (3/0/2) | +8.01 (3/0/2) |
| 4 | +177.76 (5/0/0) | -43.53 (2/0/3) | +51.49 (3/0/2) |
| 5 | +176.29 (4/0/1) | +16.11 (3/0/2) | +12.44 (3/0/2) |
| 6 | +220.82 (5/0/0) | +185.07 (4/0/1) | +89.15 (3/0/2) |
| 7 | -106.55 (0/0/5) | -95.41 (1/0/4) | -183.78 (1/1/3) |
| 8 | +21.87 (3/0/2) | -64.81 (2/0/3) | -20.85 (2/0/3) |
| 9 | +1.33 (3/0/2) | -39.79 (2/0/3) | +39.44 (3/0/2) |
| Median of bank medians | +92.26 | -33.52 | +10.22 |
| Positive banks | 6/8 | 3/8 | 5/8 |

## Interpretation

The favorable direction against classic NEAT in the earlier two-bank pilot recurred in most new banks, with substantial variation. In bank 6, Waltz beat classic NEAT on all five seeds. In bank 7, classic NEAT won all five. The eight-bank replication determines the prespecified decision. The earlier pilot is reported separately.

Four independent islands performed better than accelerating Waltz in five banks. In an exploratory direct comparison, independent islands also exceeded classic NEAT in 7 of eight banks (median of bank medians +96.99). This comparison helps interpret the single-population baseline and does not isolate a causal mechanism.

The comparison with a stationary waltzer was secondary. The modest positive overall difference and variation across banks leave the benefit of travel unresolved. This confirmation included one moving tempo schedule. Assessing the effect of acceleration itself requires comparison with another moving schedule.

## Verification

All 160 planned runs completed with 34,656,000 training steps per run. The recorded wall time was 3 h 20 min 21 s with four workers. All recorded source and protocol hashes matched. Recomputing the per-run held-out means, 120 paired differences, 24 bank summaries and three overall summaries from the raw runs reproduced the saved tables. Bank differences reflect changes in training starts, test starts and evolution seeds together.

The [prespecified protocol](pendulum_confirmation.md) records the decision rule. Full per-seed differences are in `generated/pendulum_confirmation_v1/paired_differences.csv`.

The [published result files](../results/pendulum_confirmation_v1/) include the protocol and numerical tables.
