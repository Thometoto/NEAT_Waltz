# NEAT Waltz

I got the idea for this experiment from Jacques Brel's *La Valse à mille temps*.
I wondered what would happen if a genome could visit several NEAT populations.
What would it pick up from each one? Would it help to spend more time with a
partner early on and move faster later?

In Waltz, the partner is the best genome on the island being visited. The two
genomes produce a child through crossover and mutation. If the child scores
better than the visiting parent on the training task, it takes over as the
*waltzer*. Otherwise the parent carries on. After a set number of generations,
the waltzer moves to the next island. I call that visit length the *tempo*.

Thomas Ammendola · [Read the paper](docs/NEAT_Waltz.pdf)

## Try it on Circle

From the project folder, set up Python 3.12 and run a small example:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python test_circle.py --generations 150 --seed 0 --hidden 4 --mutation-power 0.1 --waltzers 4 --tempo 5 --plot circle_waltz.png
```

Open `circle_waltz.png` to see how training error changes over the visits. The same options work
with `test_spiral.py` and `test_xor.py`. The part where a child takes over the
lineage is in [neat_waltz.py](neat_waltz.py).

## What I found

On one of the two Spiral datasets, crossing with the local champion helped the
final waltzer on all ten seeds. Compared with mutation alone, the median gain
in test accuracy was 17.19 percentage points for the waltzer and just 0.29 points
for the best genome found anywhere in the four islands. On Circle, changing
the dataset could even reverse which method did better. [Crossover results](docs/mutation_control_results.md) · [Fresh-champion comparison](docs/nonpersistent_control_results.md)

Some Waltz runs beat standard NEAT on classification. Stationary lineages and
ring migration were competitive too. Shortening visits from 15 to 10 to 5
generations gave mixed results. On Sine, where each island trained on a
different part of the curve, the waltzer struggled to fit all four parts at once.
[Tempo results](docs/tempo_pilot_results.md) · [Exchange comparison](docs/exchange_effects.png) · [Sine example](docs/sine_seed95.png)

On Pendulum, accelerating Waltz scored higher in six of eight groups of
starting states against standard NEAT and in three against four independent
islands. I also tried giving the network a signal for how much energy the
pendulum needed to swing up, along with a symmetric torque rule. All nine
Waltz v2 controllers then balanced the pendulum on all 32 test starts.
With the same help, both NEAT baselines did too. I still have an open question:
which behaviors, if any, does a waltzer keep from an earlier island?
[Pendulum comparison](docs/pendulum_confirmation_results.md) · [Visit order](results/pendulum_final_tempo_v1/banks.csv) · [Waltz v2 settings](docs/pendulum_v2.md)

## Reproduce a run

The [results folder](results/README.md) has the scores behind the paper and
instructions for replaying the 54 saved Pendulum controllers. To use Pendulum
or run the full test suite:

```bash
python -m pip install -r requirements-control.txt
python -m unittest discover -v
```

To repeat the original tempo and migration experiments, use a fresh output
folder for each command. These runs use 2,000 generations and 240,000 genome
evaluations per run and can take several hours. For a quick check, use the
first command on its own.

```bash
python confirm_tempo.py --smoke-test --workers 1 --output-dir generated/observations_smoke
python confirm_tempo.py --workers 3 --output-dir generated/tempo
python confirm_exchange.py --workers 3 --output-dir generated/exchange
```

To follow one waltzer through a run of `confirm_tempo.py`, start with
`candidate_history` in the per-run JSON. It records the parents and whether
each child took over. Use `generation_history` for the best genome across the
islands and `visits` for the host changes. Generation 0 sets up the lineages.
A visit of length 5 covers candidate evaluations 1 through 5.
[Trace fields](docs/implementation_audit.md) · [Classification pilot settings](docs/tempo_pilot_protocol.md) · [Pendulum pilot results](docs/pendulum_results.md)

Built with [NEAT-Python](https://github.com/CodeReclaimers/neat-python).
