"""Exploratory single-waltzer tempo pilot; settings are fixed before outcomes."""

import argparse
import csv
import hashlib
import json
import os
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import neat
import numpy as np
from matplotlib.figure import Figure

from diagnose_circle import classification_metrics, create_config
from neat_waltz import run_waltz
from test_circle import circle_examples
from test_spiral import spiral_examples
from test_xor import NEAT_SETTINGS
from waltz_tempo import three_phase_schedule


METHODS = ("accelerating", "constant", "decelerating", "stationary", "independent")
DATA_SEEDS = ((101, 102), (103, 104))
EVOLUTION_SEEDS = tuple(range(220, 230))
GENERATIONS = 361
SOURCES = ("pilot_tempo.py", "waltz_tempo.py", "neat_waltz.py", "neat_genome.py",
           "test_circle.py", "test_spiral.py", "test_xor.py", "diagnose_circle.py")


def visit_options(method, generations):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    options = dict(tempo=10, waltzers=0 if method == "independent" else 1,
                   travel=method != "stationary")
    if method in ("accelerating", "decelerating", "stationary"):
        tempos = (5, 10, 15) if method == "decelerating" else (15, 10, 5)
        options["tempo_schedule"], _ = three_phase_schedule(generations, tempos)
    return options


def trial(task, dataset, seed, method, generations=GENERATIONS, *, smoke=False):
    generator = {"circle": circle_examples, "spiral": spiral_examples}[task]
    train_seed, test_seed = DATA_SEEDS[dataset]
    train = generator(8 if smoke else 64, train_seed)
    test = generator(16 if smoke else 256, test_seed)
    if {x for x, _ in train} & {x for x, _ in test}:
        raise RuntimeError("Train and test inputs overlap")
    config = create_config(4, mutation_power=0.1)
    children, history, visits = [], [], []
    evaluations = 0

    def fitness(genome, cfg):
        nonlocal evaluations
        evaluations += 1
        return -classification_metrics(neat.nn.FeedForwardNetwork.create(genome, cfg), train)[0]

    started = perf_counter()
    best, final = run_waltz(
        config, fitness, generations, seed=seed, **visit_options(method, generations),
        on_candidate=lambda **e: children.append(e), on_generation=lambda **e: history.append(e),
        on_visit=lambda **e: visits.append(e),
    )
    seconds = perf_counter() - started
    if evaluations != generations * 120 or len(children) != (0 if method == "independent" else generations - 1):
        raise RuntimeError("Pilot budget mismatch")
    best_test = classification_metrics(neat.nn.FeedForwardNetwork.create(best, config), test)
    final_test = None if final is None else classification_metrics(neat.nn.FeedForwardNetwork.create(final, config), test)
    moves = sum(a["island"] != b["island"] for a, b in zip(visits, visits[1:]))
    accepted = [e for e in children if e["accepted"]]
    row = dict(task=task, dataset=dataset, seed=seed, method=method, generations=generations,
               genome_evaluations=evaluations, training_predictions=evaluations * len(train),
               candidate_evaluations=len(children), training_seconds=seconds,
               train_mse=-best.fitness, test_mse=best_test[0], test_accuracy=best_test[1],
               final_waltzer_train_mse=None if final is None else -final.fitness,
               final_waltzer_test_mse=None if final_test is None else final_test[0],
               final_waltzer_test_accuracy=None if final_test is None else final_test[1],
               moves=moves, visits=len(visits), accepted=len(accepted),
               acceptance_rate=None if not children else len(accepted) / len(children),
               accepted_gain=sum(e["fitness_gain"] for e in accepted))
    if history[-1]["genome_evaluations"] != evaluations:
        raise RuntimeError("Recorded and actual evaluation counts differ")
    return dict(metrics=row, candidate_history=children, generation_history=history, visits=visits)


def run_name(job):
    task, dataset, seed, method = job
    return f"{task}_data{dataset}_seed{seed}_{method}"


def summarize(output, results):
    """Paired pilot descriptions; dataset replications stay separate."""
    rows = [r["metrics"] for r in results]
    with (output / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    datasets = sorted({r["dataset"] for r in rows})
    figure = Figure(figsize=(6 * len(datasets), 8), layout="constrained")
    lineage_figure = Figure(figsize=(6 * len(datasets), 8), layout="constrained")
    axes = figure.subplots(2, len(datasets), squeeze=False)
    lineage_axes = lineage_figure.subplots(2, len(datasets), squeeze=False)
    for task_index, task in enumerate(("circle", "spiral")):
        for column, dataset in enumerate(datasets):
            ax = axes[task_index, column]
            lineage_ax = lineage_axes[task_index, column]
            local = [r for r in rows if r["task"] == task and r["dataset"] == dataset]
            for method in METHODS:
                traces = [r["generation_history"] for r in results
                          if r["metrics"]["task"] == task and r["metrics"]["dataset"] == dataset
                          and r["metrics"]["method"] == method]
                if traces:
                    xs = [e["genome_evaluations"] for e in traces[0]]
                    curves = [[-e["best_fitness"] for e in h] for h in traces]
                    ax.plot(xs, np.median(curves, axis=0), label=method)
                    if traces[0][0]["lineages"]:
                        lineage_curves = [[-e["lineages"][0]["fitness"] for e in h] for h in traces]
                        lineage_ax.plot(xs, np.median(lineage_curves, axis=0), label=method)
            ax.set(title=f"{task.title()} | data {dataset}", xlabel="Genome evaluations", ylabel="Median best training MSE")
            ax.legend(fontsize=8)
            lineage_ax.set(title=f"{task.title()} | data {dataset}", xlabel="Genome evaluations",
                           ylabel="Median waltzer training MSE")
            lineage_ax.legend(fontsize=8)
            phase_rounds = (local[0]["generations"] - 1) // 3
            for panel in (ax, lineage_ax):
                for boundary in (phase_rounds, 2 * phase_rounds):
                    panel.axvline((boundary + 1) * 120, color="gray", linestyle="--", linewidth=0.7)
            for method in METHODS[1:]:
                a = {r["seed"]: r for r in local if r["method"] == "accelerating"}
                b = {r["seed"]: r for r in local if r["method"] == method}
                for field in ("test_accuracy", "final_waltzer_test_accuracy"):
                    gains = [(a[s][field] - b[s][field]) * 100 for s in sorted(a.keys() & b.keys())
                             if a[s][field] is not None and b[s][field] is not None]
                    if gains:
                        summaries.append(dict(task=task, dataset=dataset, baseline=method, metric=field,
                                              n=len(gains), median_gain_pp=float(np.median(gains)),
                                              wins=sum(g > 1e-12 for g in gains),
                                              ties=sum(abs(g) <= 1e-12 for g in gains),
                                              losses=sum(g < -1e-12 for g in gains)))
    figure.suptitle("Exploratory tempo pilot | training curves, not test-based selection")
    figure.savefig(output / "training_progress.png", dpi=150)
    lineage_figure.suptitle("Exploratory tempo pilot | retained waltzer, training data only")
    lineage_figure.savefig(output / "lineage_progress.png", dpi=150)
    with (output / "paired_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("generated/tempo_pilot_v1"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--smoke-test", action="store_true", help="Small datasets and one seed/instance per task")
    parser.add_argument("--benchmark", action="store_true", help="Time the first full-size run of each method/task")
    parser.add_argument("--resume", action="store_true", help="Continue only with identical settings and source hashes")
    args = parser.parse_args()
    if args.workers < 1 or (args.smoke_test and args.benchmark):
        parser.error("Use positive workers and choose either smoke or benchmark")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    seeds = EVOLUTION_SEEDS[:1] if args.smoke_test or args.benchmark else EVOLUTION_SEEDS
    datasets = (0,) if args.smoke_test or args.benchmark else range(len(DATA_SEEDS))
    jobs = [(t, d, s, m) for t in ("circle", "spiral") for d in datasets for s in seeds for m in METHODS]
    config = create_config(4, mutation_power=0.1)
    manifest = dict(protocol="waltz_tempo_pilot_v1", stage="software_smoke" if args.smoke_test else "benchmark" if args.benchmark else "exploratory_pilot",
                    generations=GENERATIONS, evolution_seeds=list(seeds), data_seeds=DATA_SEEDS,
                    methods=METHODS, jobs=jobs, islands=4, island_population=30, waltzers=1,
                    candidates_per_waltzer=1, initial_hidden=4, mutation_power=0.1,
                    train_points=16 if args.smoke_test else 128, test_points=32 if args.smoke_test else 512,
                    genome_evaluations_per_run=GENERATIONS * 120,
                    accelerating=three_phase_schedule(GENERATIONS),
                    decelerating=three_phase_schedule(GENERATIONS, (5, 10, 15)),
                    constant_tempo=10, stationary="Same visit boundaries as accelerating, host remains 0",
                    starting_host=0, selection="Training MSE only; test outputs measured after training",
                    primary_contrasts=["accelerating vs constant", "accelerating vs decelerating"],
                    limitations="Exploratory; only two data draws per task; constant tempo does not match travel count; no migration or nonpersistent-lineage control",
                    neat_settings=NEAT_SETTINGS, python_version=platform.python_version(),
                    neat_python_version=version("neat-python"),
                    observation_schema="waltz_training_observations_v1",
                    source_sha256={n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in SOURCES})
    manifest = json.loads(json.dumps(manifest))
    output = args.output_dir
    metadata = output / "protocol.json"
    if args.resume:
        if not metadata.exists() or json.loads(metadata.read_text()) != manifest:
            parser.error("Resume requires an identical protocol, dependencies and source hashes")
    else:
        if output.exists() and any(output.iterdir()):
            parser.error("Output folder is not empty; use a fresh folder or --resume")
        output.mkdir(parents=True, exist_ok=True)
        with metadata.open("x") as stream:
            json.dump(manifest, stream, indent=2)
    results, remaining = [], []
    for job in jobs:
        path = output / f"{run_name(job)}.json"
        if path.exists():
            result = json.loads(path.read_text())
            row = result["metrics"]
            if tuple(row[k] for k in ("task", "dataset", "seed", "method")) != job or row["genome_evaluations"] != GENERATIONS * 120:
                raise RuntimeError(f"Invalid completed run: {path}")
            results.append(result)
        else:
            remaining.append(job)
    print(f"{manifest['stage']}: {len(jobs)} runs, {len(results)} already complete; {args.workers} workers.", flush=True)
    wall_start = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        futures = {pool.submit(trial, *job, smoke=args.smoke_test): job for job in remaining}
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            path = output / f"{run_name(job)}.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(result, separators=(",", ":")))
            temporary.replace(path)
            results.append(result)
            print(f"{len(results)}/{len(jobs)} | {run_name(job)} | {result['metrics']['training_seconds']:.1f}s", flush=True)
    results.sort(key=lambda r: tuple(r["metrics"][k] for k in ("task", "dataset", "seed", "method")))
    summarize(output, results)
    timing = dict(wall_seconds=perf_counter() - wall_start, workers=args.workers,
                  completed_now=len(remaining), total_completed=len(results))
    (output / "timing.json").write_text(json.dumps(timing, indent=2))
    print(f"Completed {len(results)} runs; saved scores, paired summaries and training curves to {output}", flush=True)


if __name__ == "__main__":
    main()
