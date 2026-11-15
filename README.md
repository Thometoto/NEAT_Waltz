# NEAT Waltz

**Traveling Lineages across Evolving Populations**

Thomas Ammendola

Can a lineage improve by visiting several evolving populations?

NEAT Waltz explores this idea. Several NEAT populations evolve in parallel.
One or more lineages, called **waltzers**, visit them in a fixed cycle.
At each generation, a waltzer produces an offspring through crossover with the
local champion and mutation. The offspring becomes the new waltzer if its
training fitness improves.

A **tempo** controls the length of each visit. Visits can have a fixed duration
or become shorter during training, for example from 15 to 10 to 5 generations.

## Installation

Use Python 3.12. Run these commands from the project folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Try it

Run a short Circle experiment with four waltzers and a fixed tempo of 5:

```bash
python test_circle.py --generations 150 --seed 0 --hidden 4 --mutation-power 0.1 --waltzers 4 --tempo 5 --plot circle_waltz.png
```

You can also use `test_spiral.py` or `test_xor.py` with the same options.
The main model is in [neat_waltz.py](neat_waltz.py).

## Experiments

We tested Waltz on Circle and Spiral classification, and on Sine regression
with populations specialized in different regions. Comparisons include standard
NEAT, independent islands, stationary lineages, and ring migration.

Each confirmation run uses 2,000 generations and 240,000 training genome
evaluations. The best genome is selected using training error. Test data are
used only to measure performance.

To repeat the final experiments with three processes:

```bash
python confirm_tempo.py --workers 3 --output-dir generated/tempo
python confirm_exchange.py --workers 3 --output-dir generated/exchange
```

These commands run the full comparisons and can take several hours.
Use a fresh output folder for each run.

## Results

Some Waltz configurations perform better than standard NEAT on classification.
However, the results do not show a consistent advantage of travel over stationary
lineages or ring migration. Progressively shortening visits also does not
consistently help. On Sine, waltzers do not reliably combine local improvements
into a better global solution.

The [saved results](results/README.md) include scores, settings, and datasets.
Figures: [tempo comparison](docs/paper_figures/tempo_effects.png),
[exchange comparison](docs/paper_figures/exchange_effects.png), and
[Sine example](docs/paper_figures/sine_seed95.png).

Built with [NEAT-Python](https://github.com/CodeReclaimers/neat-python).
