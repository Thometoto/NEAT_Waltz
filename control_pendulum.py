"""Compare Waltz and NEAT on continuous control at equal environment-step budgets."""

import argparse
import hashlib
import json
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from itertools import product
from math import isfinite
from multiprocessing import get_context
from pathlib import Path
from statistics import fmean
from time import perf_counter

import gymnasium as gym
import neat
import numpy as np

import pilot_tempo as pilot
from mutation_control import write_csv
from nonpersistent_control import run_nonpersistent
from test_xor import run_classic_neat

METHODS = ("classic", "independent", "stationary", "constant", "accelerating",
           "decelerating", "nonpersistent")
HORIZON = 200
GENERATIONS = {"smoke": 3, "benchmark": 11, "pilot": 361}


def episode_seeds(bank, mode):
    train = 10000 + 1000 * bank
    test = 20000 + 1000 * bank
    return (list(range(train, train + (1 if mode == "smoke" else 4))),
            list(range(test, test + (2 if mode == "smoke" else 32))))


class PendulumFitness:
    def __init__(self, env, train_seeds):
        self.env = env
        self.train_seeds = train_seeds
        self.calls = self.steps = 0

    def evaluate(self, genome, config, seeds):
        network = neat.nn.FeedForwardNetwork.create(genome, config)
        returns, steps = [], 0
        for seed in seeds:
            obs, _ = self.env.reset(seed=seed)
            total = 0.0
            for step in range(1, HORIZON + 1):
                output, = network.activate((float(obs[0]), float(obs[1]), float(obs[2]) / 8))
                if not isfinite(output):
                    raise ValueError("Nonfinite controller output")
                torque = 4 * min(1.0, max(0.0, output)) - 2
                obs, reward, terminated, truncated, _ = self.env.step(np.array([torque], dtype=np.float32))
                if terminated or truncated != (step == HORIZON):
                    raise RuntimeError("Unexpected Pendulum episode length")
                total += float(reward)
                steps += 1
            if not isfinite(total):
                raise ValueError("Nonfinite episode return")
            returns.append(total)
        return returns, steps

    def __call__(self, genome, config):
        returns, steps = self.evaluate(genome, config, self.train_seeds)
        self.calls += 1
        self.steps += steps
        return fmean(returns)


def trial(method, bank, seed, mode):
    generations = GENERATIONS[mode]
    config = pilot.create_config(4, mutation_power=0.1)
    config.genome_config.num_inputs = 3
    config.genome_config.input_keys = [-1, -2, -3]
    train, test = episode_seeds(bank, mode)
    history, children, visits = [], [], []
    # Short runs use a prefix of the same calendar as the full pilot.
    options = None if method == "classic" else pilot.visit_options(
        "accelerating" if method == "nonpersistent" else method, GENERATIONS["pilot"])
    with gym.make("Pendulum-v1", g=10.0) as env:
        fitness = PendulumFitness(env, train)
        start = perf_counter()
        if method == "classic":
            best, config = run_classic_neat(config, fitness, generations, seed,
                                             on_generation=lambda **e: history.append(e))
            secondary = None
        elif method == "nonpersistent":
            result = run_nonpersistent(config, fitness, generations, seed=seed,
                                       tempo_schedule=options["tempo_schedule"])
            best, secondary = result["best"], result["stream_best"]
            history, children, visits = (result["generation_history"], result["candidate_history"],
                                         result["visits"])
        else:
            best, secondary = pilot.run_waltz(config, fitness, generations, seed=seed, **options,
                                              on_generation=lambda **e: history.append(e),
                                              on_candidate=lambda **e: children.append(e),
                                              on_visit=lambda **e: visits.append(e))
        seconds = perf_counter() - start
        expected = 120 * generations
        expected_children = 0 if method in ("classic", "independent") else generations - 1
        if (fitness.calls != expected or fitness.steps != expected * len(train) * HORIZON
                or len(children) != expected_children or len(history) != generations
                or history[-1]["best_fitness"] != best.fitness):
            raise RuntimeError("Training budget or champion mismatch")
        returns, test_steps = fitness.evaluate(best, config, test)
        secondary_returns, secondary_steps = ([], 0) if secondary is None else fitness.evaluate(secondary, config, test)
    kind = None
    if secondary is not None:
        kind = "diagnostic_stream_archive" if method == "nonpersistent" else "final_waltzer"
    metrics = dict(method=method, bank=bank, seed=seed, mode=mode, generations=generations,
                   genome_evaluations=fitness.calls, candidate_evaluations=len(children),
                   training_episodes=fitness.calls * len(train), training_steps=fitness.steps,
                   heldout_steps=test_steps + secondary_steps, training_seconds=seconds,
                   best_train_return=best.fitness, best_test_return=fmean(returns), secondary_kind=kind,
                   secondary_train_return=secondary.fitness if secondary else None,
                   secondary_test_return=fmean(secondary_returns) if secondary_returns else None)
    return dict(metrics=metrics, train_seeds=train, test_seeds=test, best_episode_returns=returns,
                secondary_episode_returns=secondary_returns, generation_history=history,
                candidate_history=children, visits=visits)


def save_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=GENERATIONS, default="smoke")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    mode = args.mode
    methods = ("accelerating",) if mode == "benchmark" else METHODS
    banks = range(2 if mode == "pilot" else 1)
    seeds = range(300, 310 if mode == "pilot" else 301)
    jobs = list(product(methods, banks, seeds))
    out = args.output_dir or Path("generated") / f"pendulum_{mode}_v1"
    sources = (*pilot.SOURCES, "control_pendulum.py", "nonpersistent_control.py", "mutation_control.py")
    manifest = dict(mode=mode, generations=GENERATIONS[mode], methods=list(methods),
                    evolution_seeds=list(seeds), episode_banks=[episode_seeds(b, mode) for b in banks],
                    environment="Pendulum-v1", gravity=10, horizon=HORIZON, total_genomes=120,
                    primary="Held-out mean return of the best genome selected on training return",
                    python=platform.python_version(), versions={k: version(k) for k in
                    ("gymnasium", "neat-python", "numpy", "matplotlib")}, source_sha256={
                    p: hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest() for p in sources})
    # JSON round-trip normalizes tuples for the resume comparison.
    manifest = json.loads(json.dumps(manifest))
    out.mkdir(parents=True, exist_ok=True)
    protocol = out / "protocol.json"
    if any(out.iterdir()):
        if not args.resume or not protocol.is_file() or json.loads(protocol.read_text()) != manifest:
            parser.error("Use a fresh output directory or --resume with a matching protocol")
    else:
        save_json(protocol, manifest)
    rows, pending = [], {}
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        for method, bank, seed in jobs:
            path = out / f"{method}_bank{bank}_seed{seed}.json"
            if path.exists():
                metrics = json.loads(path.read_text())["metrics"]
                if (metrics["method"], metrics["bank"], metrics["seed"], metrics["mode"]) != (method, bank, seed, mode):
                    raise RuntimeError(f"Run identity mismatch in {path}")
                rows.append(metrics)
            else:
                pending[pool.submit(trial, method, bank, seed, mode)] = path
        for future in as_completed(pending):
            result = future.result()
            save_json(pending[future], result)
            rows.append(result["metrics"])
            print(f"{len(rows)}/{len(jobs)} {pending[future].stem}", flush=True)
    rows.sort(key=lambda r: (r["bank"], r["seed"], r["method"]))
    write_csv(out / "scores.csv", rows)
    if pending:
        save_json(out / "timing.json", dict(session_seconds=perf_counter() - start,
                  newly_completed=len(pending), workers=args.workers, total_runs=len(rows)))


if __name__ == "__main__":
    main()
