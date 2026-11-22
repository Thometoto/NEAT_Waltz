# Swing-up and balance with NEAT Waltz

The v2 experiment targets a behavioral goal. A successful episode spends at least 45 of its last 50 steps within 15 degrees of upright with an angular speed of at most 1 rad/s. A controller meets the study target when at least 29 of its 32 held-out episodes succeed. These are project criteria. Evolution still maximizes Gymnasium's original mean episode return.

## What changes

The three original measurements are cosine and sine of the angle, and angular velocity. The energy policy computes a deficit to the upright resting state, in normalized mechanical units, as `15 × (1 − cos θ) − ω² / 2`. The coefficient uses the gravity and inertia of this particular simulator. During swing-up the three network inputs are `(tanh(8ω) × tanh(deficit/2), 0, 0)`. Within 0.5 rad of the top and below 3 rad/s, they switch to `(0, 3 sin θ, ω)` to allow angle and velocity feedback. This is substantial task-specific prior knowledge. The network still determines the torque. It receives both the feature vector and its reflection in two forward passes. The torque is twice the difference of the two sigmoid outputs. The resulting controller has zero torque at upright rest, bounded torque, and equal and opposite responses in mirrored states.

V2 starts one lineage at each of the four island champions. Each lineage travels cyclically. At each generation its host supplies four population slots, one for a standard champion crossover followed by NEAT mutation and three for copies of the current waltzer. Each copy retains its topology and perturbs up to two enabled connection weights or node biases. The three Gaussian perturbation scales are 0.08, 0.35 and 1.0. The highest training-scoring child can replace its lineage parent on strict improvement. The other population slots use the project's ordinary NEAT reproduction. Sixteen candidates per generation consume slots inside the common population of 120. The deployed output is the best training-selected genome seen across all islands.

## Comparison

All methods use the same 120 genomes in total, initial four hidden nodes, ordinary NEAT configuration, training starts, held-out starts, 200-step horizon and original reward. Two baselines use the original observations and torque mapping. Two more use the energy features and symmetric mapping with classic NEAT or four independent islands. Single-lineage Waltz uses the same energy policy and one candidate per generation. These controls separate the effects of the shared control policy from those of the combined reproduction changes. Matched environment steps do not imply matched network forward calls or CPU time.

The software/development stage uses one separate bank and seven evaluated generations. The evaluation uses three fresh banks and three new evolution seeds per bank. All six methods use 73 evaluated generations, 8,760 genome evaluations, and 7,008,000 training steps per run. The two traveling methods visit islands for 3, 2 and 1 rounds, respectively, in three 24-round phases. This is a shorter schedule and budget than the historical 361-generation experiment. The four training starts are fixed within each bank. Each champion is scored on 32 disjoint starts only after selection. Compare paired held-out return differences within each bank. Many episodes share a controller and a common bank, so counts of successful episodes describe behavior rather than independent algorithm replications.

## Reproduce

From the repository root with the control requirements installed:

```sh
.venv/bin/python pendulum_waltz_v2.py --stage development --workers 4
.venv/bin/python pendulum_waltz_v2.py --stage evaluation --workers 4
.venv/bin/python replay_pendulum_v2.py generated/pendulum_waltz_v2_evaluation/waltz2_energy_b1_s1203.json --output generated/pendulum_waltz_v2_evaluation/replay.html
```

Each run saves the selected feedforward network, seeds, episode returns and stability flags. The replay verifies that the saved controller reproduces all 32 episode scores and flags in Gymnasium. Output directories are under `generated/`. Run one instance of each stage at a time.
