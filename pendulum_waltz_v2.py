"""Pendulum experiment with energy features and local parameter search around traveling lineages."""

import argparse
import hashlib
import json
import math
import platform
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import copy, deepcopy
from importlib.metadata import version
from pathlib import Path
from statistics import fmean, median

import gymnasium as gym
import neat
import numpy as np

import control_pendulum as original
from neat_waltz import (_create_populations, _evaluate_champion, _insert_waltzer_candidate,
                        _replacement_priority, _reproduce)
from waltz_tempo import three_phase_schedule

ROOT = Path(__file__).resolve().parent
METHODS = ("classic_raw", "independent_raw", "classic_energy", "independent_energy",
           "waltz1_energy", "waltz2_energy")
CATCH_COS = math.cos(0.5)
UPRIGHT_COS = math.cos(math.radians(15))


def features(obs):
    cosine, sine, velocity = map(float, obs)
    if cosine > CATCH_COS and abs(velocity) < 3.0:
        return (0., 3 * sine, velocity)
    deficit = 15 * (1 - cosine) - 0.5 * velocity**2
    return (math.tanh(8 * velocity) * math.tanh(deficit / 2), 0., 0.)


def action(network, obs, energy):
    if energy:
        x = features(obs)
        # Opposite states should generate opposite torques, including zero at rest.
        a, b, c = x
        return 2 * (network.activate(x)[0] - network.activate((-a, -b, -c))[0])
    c, s, v = map(float, obs)
    return 4 * network.activate((c, s, v / 8))[0] - 2


def evaluate(env, network, seeds, energy, *, assess=False):
    returns, successes = [], []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        total = 0.
        stable = 0
        for step in range(200):
            torque = action(network, obs, energy)
            obs, reward, done, truncated, _ = env.step(np.array([torque], dtype=np.float32))
            if done or truncated != (step == 199):
                raise RuntimeError("Unexpected Pendulum horizon")
            total += float(reward)
            if assess and step >= 150 and obs[0] >= UPRIGHT_COS and abs(obs[2]) <= 1:
                stable += 1
        returns.append(total)
        if assess:
            successes.append(stable >= 45)
    return returns, successes


def local_child(island, offspring, parent, slot, sigma):
    child = deepcopy(parent)
    child.key = next(island.reproduction.genome_indexer)
    child.fitness = None
    cfg = island.config.genome_config
    parameters = [(c, "weight", cfg.weight_min_value, cfg.weight_max_value)
                  for c in child.connections.values() if c.enabled]
    parameters.extend((n, "bias", cfg.bias_min_value, cfg.bias_max_value) for n in child.nodes.values())
    for gene, attr, low, high in random.sample(parameters, min(2, len(parameters))):
        value = getattr(gene, attr) + random.gauss(0, sigma)
        setattr(gene, attr, max(low, min(high, value)))
    del offspring[slot.key]
    offspring[child.key] = child
    return child


def run_refined(config, fitness, generations, seed, schedule):
    """Four lineages each breed one crossover and three weight/bias variants per round."""
    config = deepcopy(config)
    rng = random.getstate()
    try:
        random.seed(seed)
        config.seed = None
        islands = _create_populations(config, 4)
        groups, lineages = [], []
        best = None
        visits, remaining = 0, schedule[0]
        for generation in range(generations):
            if any(len(p.population) != config.pop_size for p in islands):
                raise RuntimeError("Island size changed")
            champions = [_evaluate_champion(p, fitness) for p in islands]
            current = max(champions, key=lambda g: g.fitness)
            if best is None or current.fitness > best.fitness:
                best = deepcopy(current)
            if generation == 0:
                lineages = list(map(deepcopy, champions))
            else:
                for i, children in enumerate(groups):
                    candidate = max(children, key=lambda g: g.fitness)
                    if candidate.fitness > lineages[i].fitness:
                        lineages[i] = deepcopy(candidate)
                remaining -= 1
                if remaining == 0:
                    visits += 1
                    remaining = schedule[min(visits // 4, len(schedule) - 1)]
            if generation + 1 == generations:
                break
            groups = [[] for _ in islands]
            for host, island in enumerate(islands):
                i = (host - visits) % 4
                parent = lineages[i]
                offspring = _reproduce(island, generation)
                slots = sorted(offspring.values(), key=_replacement_priority)[:4]
                groups[i].append(_insert_waltzer_candidate(island, offspring, parent, champions[host], slot=slots[0]))
                for slot, sigma in zip(slots[1:], (0.08, 0.35, 1.0)):
                    groups[i].append(local_child(island, offspring, parent, slot, sigma))
                island.population = offspring
                island.species.speciate(config, offspring, generation)
        return best
    finally:
        random.setstate(rng)


def network_record(network):
    return dict(inputs=network.input_nodes, outputs=network.output_nodes,
                nodes=[dict(id=n, activation=act.__name__, aggregation=agg.__name__, bias=b,
                            response=r, links=links) for n,act,agg,b,r,links in network.node_evals])


def trial(method, bank, seed, generations):
    energy = method.endswith("_energy")
    train = list(range(120000 + 1000*bank, 120004 + 1000*bank))
    test = list(range(220000 + 1000*bank, 220032 + 1000*bank))
    config = original.pilot.create_config(4, mutation_power=.1)
    config.genome_config.num_inputs = 3
    config.genome_config.input_keys = [-1,-2,-3]
    if generations == 361:
        schedule, _ = three_phase_schedule(361)
    else:
        schedule, _ = three_phase_schedule(73, (3,2,1))
    calls = 0
    with gym.make("Pendulum-v1", g=10.) as env:
        def fitness(genome, cfg):
            nonlocal calls
            calls += 1
            net = neat.nn.FeedForwardNetwork.create(genome, cfg)
            returns, _ = evaluate(env, net, train, energy)
            return fmean(returns)

        if method.startswith("classic_"):
            best, config = original.run_classic_neat(config, fitness, generations, seed)
        elif method == "waltz2_energy":
            best = run_refined(config, fitness, generations, seed, schedule)
        else:
            best, _ = original.pilot.run_waltz(config, fitness, generations, seed=seed,
                                               tempo_schedule=schedule,
                                               waltzers=0 if method.startswith("independent_") else 1)
        if calls != 120 * generations:
            raise RuntimeError("Budget mismatch")
        network = neat.nn.FeedForwardNetwork.create(best, config)
        returns, successes = evaluate(env, network, test, energy, assess=True)
    return dict(method=method, bank=bank, seed=seed, generations=generations,
                train_seeds=train, test_seeds=test, evaluations=calls, training_steps=calls*800,
                train_return=best.fitness, mean_return=fmean(returns), episode_returns=returns,
                successes=sum(successes), episode_success=successes, network=network_record(network))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("development","evaluation"), default="development")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    dev = args.stage == "development"
    generations = 7 if dev else 73
    pairs = [(0, 1100)] if dev else [(b, 1200+3*b+i) for b in range(1,4) for i in range(3)]
    jobs = [(m,b,s) for b,s in pairs for m in METHODS]
    out = ROOT / "generated" / f"pendulum_waltz_v2_{args.stage}"
    out.mkdir(exist_ok=True,parents=True)
    files = (*original.pilot.SOURCES, "control_pendulum.py", "pendulum_waltz_v2.py")
    protocol = dict(stage=args.stage, methods=list(METHODS), pairs=list(map(list,pairs)),
                    generations=generations, workers_independent=True, python=platform.python_version(),
                    versions={p:version(p) for p in ("gymnasium","neat-python","numpy")},
                    source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
                    success="At least 45 of the final 50 states within 15 degrees of upright and speed <= 1 rad/s",
                    solved_controller="At least 29 of 32 held-out episodes succeed",
                    primary="Paired mean held-out return differences, aggregated separately by bank")
    path=out/"protocol.json"
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise ValueError("Existing results have a different protocol")
    original.save_json(path,protocol)
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={}
        for method,bank,seed in jobs:
            path=out/f"{method}_b{bank}_s{seed}.json"
            if path.exists():
                row=json.loads(path.read_text())
                if (row["method"],row["bank"],row["seed"],row["generations"])!=(method,bank,seed,generations):
                    raise ValueError(f"Unexpected trial at {path}")
                rows.append(row)
            else:
                pending[pool.submit(trial,method,bank,seed,generations)]=path
        for future in as_completed(pending):
            row=future.result()
            original.save_json(pending[future],row)
            rows.append(row)
            print(f"{len(rows)}/{len(jobs)} {pending[future].stem}",flush=True)
    cols=("method","bank","seed","train_return","mean_return","successes","training_steps")
    original.write_csv(out/"scores.csv",[{k:r[k] for k in cols} for r in sorted(rows,key=lambda r:(r['bank'],r['seed'],r['method']))])
    for method in METHODS:
        group=[r for r in rows if r['method']==method]
        print(method, round(median(r['mean_return'] for r in group),2),
              [r['successes'] for r in group])


if __name__=="__main__":
    main()
