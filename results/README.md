# Published results

These files contain the final numerical results used in the article. They are
copied without modifying the original experimental outputs.

## Tempo confirmation

`waltz_tempo_confirmation/` contains Circle, Spiral, and Sine results for six
methods and evolution seeds 95–104: 60 trials per task, 180 in total.

For each task:

- `TASK.csv`: individual trial scores and measured evaluation budgets.
- `TASK.summary.csv`: paired comparisons, medians, win/tie/loss counts, and
  pointwise 95% bootstrap intervals based on 10,000 resamples.
- `TASK.json`: protocol, exact training and test examples, settings, and source
  hashes recorded by the experiment.

The six `sine_seed95_METHOD.json` files contain the curves and histories needed
to rebuild the Sine figure. Seed 95 is the first confirmation seed, selected
without ranking its test performance. Other per-run traces and previews are
omitted to keep this repository small; the experiment scripts regenerate them.

Circle and Spiral use training/test data seeds 17/18 and 512/256 examples.
Sine uses data seeds 22/23 and 512/1,024 examples.

## Exchange confirmation

`waltz_exchange_confirmation/` uses the same three file types for Circle and
Spiral. Five methods and evolution seeds 75–94 give 100 trials per task, 200 in
total. The data seeds are 15/16, with 512 training and 256 test examples.

This series compares fixed-5 travel against standard NEAT, independent islands,
stationary lineages, and champion ring migration. Its datasets differ from the
tempo comparison; analyze the two series separately.

## Figures and provenance

The three article figures are stored once, as PNG images in
`../docs/`: `tempo_effects.png`, `exchange_effects.png`, and
`sine_seed95.png`. `../plot_paper.py` can regenerate PDF and PNG versions from
the saved numerical data and checks the paired statistics against the saved
summaries.

The protocol JSON files retain their original source hashes. Every source file
named by those hashes is included at the repository root. Use the recorded hashes to identify the source versions used for each experiment. Historical output paths in
the records refer to the original experiment; they need not exist in this
smaller distribution.

Test data were used for reporting only. For the classification and regression experiments, global training MSE determined the primary output. Bootstrap intervals describe variation across evolution
seeds on each fixed dataset; they are not corrected for multiple comparisons.

## Follow-up experiments

Each directory below contains the frozen protocol and compact result files from the corresponding run. Full training traces are generated locally.

| Experiment | Published records |
| --- | --- |
| [Classification tempo pilot](tempo_pilot_v1/) | Per-run scores and paired summaries for Circle and Spiral |
| [Mutation-only control](mutation_control_v1/) | Scores, within-seed differences and paired summaries |
| [Nonpersistent crossover](nonpersistent_control_v1/) | Scores, within-seed differences and paired summaries |
| [Original Pendulum pilot](pendulum_pilot_v1/) | Scores on the two pilot groups |
| [Pendulum confirmation](pendulum_confirmation_v1/) | Scores, paired differences, eight group summaries and overall comparisons |
| [Pendulum visit order](pendulum_final_tempo_v1/) | Accelerating and decelerating returns for each matched pair and group summaries |
| [Physics-informed Pendulum](pendulum_waltz_v2/) | Verified episode counts, returns by controller, paired group summaries, and a ZIP with all 54 selected controllers and the protocol |

To replay the 54 controllers from a fresh checkout, install `requirements-control.txt`, then run the following commands from the repository root.

```sh
python -m zipfile -e results/pendulum_waltz_v2/verified_controllers.zip generated/pendulum_waltz_v2_evaluation
python verify_pendulum_v2.py
```

The verifier replays 216 training episodes and 1,728 held-out episodes, checks scores and stability flags, and rebuilds the CSV summaries. To generate an interactive replay of one controller, run the command in the [Pendulum protocol](../docs/pendulum_v2.md).

To retrain the separate visit-order comparison, run `python confirm_pendulum.py --workers 4 --resume` followed by `python last_pendulum_tempo.py --workers 4`. The second runner checks the first campaign's protocol before starting.
