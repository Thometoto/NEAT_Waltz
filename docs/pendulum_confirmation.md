# Independent replication on Pendulum-v1

The pilot favored accelerating Waltz over classic NEAT on both initial-state banks. The difference relative to independent islands changed sign between banks. This replication evaluates the same algorithms and budgets on eight new banks. Training, selection and evaluation reuse the original Pendulum trial function.

## Design

For each bank numbered 2 through 9, use the pilot reset-seed formula. Each bank has four fixed training starts beginning at 10000 + 1000 × bank and 32 held-out starts beginning at 20000 + 1000 × bank. Bank 2 uses evolution seeds 400–404, bank 3 uses 405–409, continuing to 435–439 for bank 9. Within each bank all four methods share its five evolution seeds and reset states. Every bank contributes equally. The pilot records and software-test records are excluded from the analysis.

Each of the 160 runs uses 361 evaluated generations of 120 genomes in total. The common budget is 34,656,000 training steps per run. Acceleration follows the original 15, 10 and 5 generation visits. The stationary lineage has identical breeding settings and stays on its starting island.

## Prespecified analysis

Select the champion by training return over all populations and generations. Measure its mean return on the bank's 32 held-out starts. For each comparator, subtract its return from that of accelerating Waltz at the same bank and evolution seed. Report all paired differences, each bank's median and win/tie/loss counts, and the median of the eight bank medians. Also report the number of banks with a strictly positive median. Keep ties at zero. The 32 episodes within a run share a controller.

Evaluate two questions in order.

1. Does the pilot advantage over classic NEAT recur across new banks?
2. Does adding the Waltz breeding scheme to four islands improve on four independently evolving islands?

For each question, define a reproducible directional gain in this campaign as a strictly positive paired median in at least seven of eight banks. This threshold is a descriptive decision criterion. Report the size and spread of gains even when the criterion is met. Meeting only the first criterion supports a comparison against the single-population baseline and leaves the added value over independent islands open. Report the stationary control as a secondary comparison of travel under the shared breeding policy. This design does not compare visit tempos.

Complete all eight banks before interpreting the comparisons. Use the final waltzer for secondary descriptive inspection only. Each bank changes training and test starts together, so differences between banks cannot locate the cause in one set alone.

## Run locally on macOS

From the repository root, run `caffeinate -i .venv/bin/python confirm_pendulum.py --workers 4 --resume`. Completed runs are reused after interruption. Read `summary.csv`, `banks.csv` and `paired_differences.csv` in `generated/pendulum_confirmation_v1/`. The dedicated `--smoke` workflow uses bank 90, evolution seed 950 and separate outputs.
