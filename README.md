# NEAT Waltz

**Traveling Lineages across Evolving Populations**  
Thomas Ammendola

NEAT Waltz explores whether a lineage can improve by visiting several evolving
populations. One or more persistent lineages, called **waltzers**, move between
NEAT populations in a fixed cycle. Each waltzer produces a candidate through
crossover with its host population's champion and mutation. The candidate takes
over the traveling role only if its training fitness improves.

A **tempo** controls the number of generations spent in each population.
Shorter visits mean faster travel. Candidates occupy existing population slots,
so the total number of genome evaluations stays the same across methods.

This repository contains the implementation, the experiments used in the
article, and their final numerical results. It uses
[NEAT-Python](https://github.com/CodeReclaimers/neat-python).

## Installation

The experiments used Python **3.12.15**, NEAT-Python **2.0.0**, NumPy **2.5.4**,
and Matplotlib **3.11.2**. Run the commands below from this directory.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in
PowerShell instead of the `source` command.

## Try the model

Run a short Circle experiment and save a graph of the waltzers' progress:

```bash
python test_circle.py --generations 150 --seed 0 --hidden 4 --mutation-power 0.1 --waltzers 4 --tempo 5 --plot circle_waltz.png
```

The same options work with `test_spiral.py` and `test_xor.py`. These files are
experiment scripts, despite their `test_` names. Their historical defaults differ
from the article settings; use the confirmation commands below for those results.

To use Waltz with another objective:

```python
from neat_waltz import run_waltz

# config: a NEAT-Python configuration; config.pop_size is the size of ONE island.
# fitness(genome, config): a deterministic, finite score to maximize.
best, final_waltzer = run_waltz(
    config, fitness, generations=2000,
    islands=4, waltzers=4, tempo=5, seed=0,
)
```

With four islands of 30 genomes, this evaluates exactly `2000 * 4 * 30` genomes,
including initialization. `best` is selected by training fitness across the run;
`final_waltzer` is the best retained final lineage by the same criterion.
`neat_genome.py` provides homologous initial hidden-node identifiers.

For progressive visits, `balanced_schedule` in `waltz_tempo.py` returns the
schedule used in the final experiment. At 2,000 generations, candidate generations
1–660 use visits of 15 generations, 661–1340 use 10, and 1341–1999 use 5.
Generation 0 initializes the lineages. Pass the returned schedule as
`tempo_schedule=` to `run_waltz`. Passing `(15, 10, 5)` directly uses only one
complete tour at each value before retaining 5; it is a different schedule.

## Reproduce the article experiments

The saved results are already included. The following commands start new training
runs in separate folders; they do not overwrite the published results.

```bash
python confirm_tempo.py --workers 3 --output-dir generated/tempo
python confirm_exchange.py --workers 3 --output-dir generated/exchange
```

`--workers 3` runs up to three independent trials at once. A full confirmation
can take several hours. Use `--task circle`, `--task spiral`, or, for the tempo
comparison, `--task sine` to run one task.

| Series | Tasks | Evolution seeds | Methods | Runs |
| --- | --- | --- | --- | --- |
| Tempo confirmation | Circle, Spiral, Sine | 95–104 | NEAT, independent islands, fixed 5, fixed 15, progressive, stationary lineages | 180 |
| Exchange confirmation | Circle, Spiral | 75–94 | NEAT, independent islands, fixed-5 travel, stationary lineages, ring migration | 200 |

Each run uses 2,000 evaluated generations and 240,000 training genome evaluations.
All methods evaluate 512 training points per genome. Standard NEAT has one
population of 120; the other methods use four islands of 30. Networks start with
four hidden nodes and `full_direct` connections. Weight and bias mutation use a
Gaussian perturbation standard deviation of 0.1.

Circle and Spiral use four waltzers and one candidate per waltzer per generation
after initialization. Sine uses one waltzer. Its ordinary island populations are
selected on four different regions of the function. Lineage acceptance and final
output selection use global training MSE. This is a centralized benchmark.

The primary output is the best genome seen by global training MSE. Test data do
not guide reproduction, lineage acceptance, or output selection. Spiral test
accuracy measures classification on the two sampled arms, not between them.

For a short installation check instead of a full experiment:

```bash
python confirm_tempo.py --smoke-test --workers 1 --output-dir generated/smoke_tempo
python confirm_exchange.py --smoke-test --workers 1 --output-dir generated/smoke_exchange
```

## Results

Median test scores in the final tempo comparison, over ten evolution seeds:

| Method | Circle accuracy (%) | Spiral accuracy (%) | Sine MSE |
| --- | ---: | ---: | ---: |
| Standard NEAT | 83.20 | 83.98 | 0.01334 |
| Independent islands | 86.72 | 90.04 | 0.01588 |
| Waltz, fixed 5 | 91.99 | 100.00 | 0.01903 |
| Waltz, fixed 15 | 93.16 | 97.07 | 0.01636 |
| Waltz, progressive | 91.41 | 93.75 | 0.01732 |
| Stationary lineages | 92.97 | 96.48 | 0.01912 |

Several Waltz configurations outperform standard NEAT on these classification
tasks. The comparisons do not establish a consistent advantage of travel over
stationary lineages or ring migration, or of progressive visits over fixed visits.
On Sine, traveling lineages do not reliably combine local improvements into a
better global solution.

The two confirmation series use different fixed datasets. Their scores should
not be pooled as if they came from one experiment. Reported intervals describe
variation across evolution seeds for those datasets. They are pointwise bootstrap
intervals and are not corrected for multiple comparisons.

Article figures: [tempo comparison](docs/paper_figures/tempo_effects.png),
[exchange comparison](docs/paper_figures/exchange_effects.png), and
[Sine example](docs/paper_figures/sine_seed95.png).

## Files

| Files | Purpose |
| --- | --- |
| `neat_waltz.py`, `neat_genome.py` | Waltz and homologous initialization |
| `neat_waltz_domains.py`, `waltz_tempo.py` | Region-specialized populations and progressive schedule |
| `neat_migration.py` | Ring-migration control |
| `confirm_tempo.py`, `confirm_exchange.py` | Final experiment entry points |
| `test_*.py`, `compare_models.py`, `diagnose_circle.py`, `preview_spiral.py`, `plot_results.py` | Experiment and plotting functions used by the entry points |
| `confirm_waltz.py` | Earlier experiment entry point, retained because experiment provenance checks require its source file |
| `plot_paper.py` | Rebuild the three article figures from saved results |
| `results/` | Final scores, protocols, and selected Sine traces |
| `docs/paper_figures/` | Three article figures in PNG format |

Development tests, exploratory results, environments, and historical release
copies are excluded. The research implementation is preserved without renaming
its modules, so recorded source hashes remain valid.

See [results/README.md](results/README.md) for the data layout. To rebuild the
article figures without training or overwriting the supplied images:

```bash
python - <<'PY'
from pathlib import Path
import plot_paper

plot_paper.OUTPUT = Path("generated/paper_figures")
for path in plot_paper.build_figures():
    print(path)
PY
```

Choose a fresh output directory when repeating a command. The scripts protect
existing results and figures against accidental overwriting.
