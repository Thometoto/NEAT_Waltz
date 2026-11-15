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
`../docs/paper_figures/`: `tempo_effects.png`, `exchange_effects.png`, and
`sine_seed95.png`. `../plot_paper.py` can regenerate PDF and PNG versions from
the saved numerical data and checks the paired statistics against the saved
summaries.

The protocol JSON files retain their original source hashes. Every source file
named by those hashes is included at the repository root. The Python sources
are unchanged from `NEAT_Models` at preparation time. Historical output paths in
the records refer to the original experiment; they need not exist in this
smaller distribution.

Test data were used for reporting only. The primary output was selected by
global training MSE. Bootstrap intervals describe variation across evolution
seeds on each fixed dataset; they are not corrected for multiple comparisons.
