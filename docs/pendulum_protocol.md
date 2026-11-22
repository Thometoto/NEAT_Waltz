# Waltz on Pendulum-v1

## Research question

Does accelerating Waltz find controllers that generalize to new starting states better than standard NEAT at the same evaluation budget? A controller chooses a torque at each step and its choice changes the next observation. This adds a sequential decision task to the classification experiments.

In the preceding tempo pilot, reversing the visit schedule changed the direction of the best-genome accuracy difference between data draws on both Circle and Spiral. The strongest crossover result concerned the final waltzer on one Spiral draw (+17.19 percentage points versus mutation alone). On that same draw, the best genome found anywhere gained only 0.29 points. Replacing the persistent parent with a fresh champion also gave mixed effects, including opposite median differences on the two Circle draws. These experiments motivate measuring the two outputs separately. The 40 accelerating runs serve as shared references across the ablations.

[Tempo results](tempo_pilot_results.md) · [Mutation control](mutation_control_results.md) · [Fresh-champion control](nonpersistent_control_results.md)

## Experiment

Use Gymnasium 1.3.0 with standard gravity 10 and a 200-step horizon. The network receives the cosine and sine of the angle and the angular velocity divided by eight. Its sigmoid output maps linearly to torque in the range −2 to +2. Fitness is the mean undiscounted episode return using the environment's reward. Higher is better.

[Gymnasium Pendulum specification](https://gymnasium.farama.org/v1.3.0/environments/classic_control/pendulum/)

| Method | Population and breeding rule |
| --- | --- |
| Classic NEAT | One population of 120 with the project's shared NEAT settings |
| Independent | Four islands of 30 evolving independently |
| Stationary | The retained lineage breeds on island 0 throughout |
| Constant | The lineage changes islands every 10 candidate rounds |
| Accelerating | Visit lengths decrease from 15 to 10 to 5 |
| Decelerating | Visit lengths increase from 5 to 10 to 15 |
| Nonpersistent | Fresh champions from the host and preceding island breed on the accelerating calendar |

The three phases last 120 candidate rounds each after generation 0. Acceleration and deceleration each produce 43 moves. Constant tempo produces 35. Each method with candidate injection evaluates one such child per round within the common population budget. Every controller starts evolution under the same configuration of four initial hidden nodes and weight and bias mutation power 0.1.

| Bank | Training reset seeds | Held-out reset seeds | Evolution seeds |
| --- | --- | --- | --- |
| 0 | 10000–10003 | 20000–20031 | 300–309 |
| 1 | 11000–11003 | 21000–21031 | 300–309 |

All genomes in a run face the same four training starts throughout evolution, so the stored score of a retained parent stays comparable with later offspring. This experimental choice allows specialization to those starts. Each trained champion is therefore evaluated on 32 held-out starts. All methods within a bank share both sets.

The pilot contains 140 runs, each with 361 evaluated generations × 120 genomes × 4 starts × 200 steps, or 34,656,000 training steps. Held-out evaluations occur after training and have separate counters.

## Analysis

The primary comparison pairs accelerating Waltz with classic NEAT by bank and evolution seed. For each run, select the highest training-scoring genome from all populations and generations. Report its mean return over the 32 held-out starts. Compute the paired difference (accelerating minus classic) for each evolution seed and report all ten differences, their median, and win/tie/loss counts separately by bank. The 32 episodes estimate a controller's performance and the evolutionary run is the replication unit.

Compare acceleration with deceleration to test visit order at matched move count. Compare stationary and independent islands to interpret what the moving lineage adds. For each lineage method, also report the final waltzer's mean held-out return as a secondary outcome. For nonpersistent crossover, the corresponding diagnostic is the best training-scored member of its candidate stream, including the initial host champion. Its archive never breeds.

## Reproduction

From the repository root, using the project Python environment:

```sh
python -m pip install -r requirements-control.txt
python -m unittest discover -v
python control_pendulum.py --mode smoke --workers 3
python control_pendulum.py --mode benchmark --workers 1
python control_pendulum.py --mode pilot --workers 3
```

Each mode writes its own directory under `generated/` with the source hashes, package versions, reset seeds, generation and child traces, per-episode held-out returns, scores and timing. Use `--resume` to reuse completed runs under the same protocol. Smoke uses three generations, one training start and two held-out starts for each of seven methods. Benchmark uses eleven generations of accelerating Waltz with the full episode counts. Both use the initial prefix of the pilot calendar. Timing and software checks are excluded from the scientific comparison.

## Timing check on 20 November 2026

All 15 existing unit tests passed. The seven short runs completed 504,000 training steps in total. Each generation had the expected population sizes and cumulative evaluation count. Initial island champion scores matched across the six island methods. Each held-out episode consumed 200 steps.

The single-worker benchmark completed 1,320 genome evaluations and 1,056,000 training steps in 7.66 seconds of training (8.00 seconds including startup, held-out evaluation and output). Applying that training rate to 140 runs gives 9.77 hours serially or 3.26 hours under ideal three-worker scaling. These projections assume later generations and all methods have the same per-step cost as the first eleven generations of accelerating Waltz. Evolving network size and sustained parallel throughput may change the actual duration. The [completed pilot results](pendulum_results.md) give the observed runtime and the prespecified comparisons.
