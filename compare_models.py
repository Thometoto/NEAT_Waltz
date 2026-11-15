"""Compare NEAT methods and save reproducible scores and boundary previews."""

import argparse
import csv
import hashlib
import json
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import version
from math import isfinite
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import neat

from diagnose_circle import classification_metrics, create_config, save_diagnostic
from neat_genome import INITIALIZATION
from neat_migration import run_ring_migration
from neat_waltz import run_waltz
from plot_results import METHODS, save_plots
from test_circle import circle_examples
from test_spiral import spiral_examples
from test_xor import NEAT_SETTINGS, XOR, run_classic_neat


FIELDS = ("generations", "seed", "method", "genome_evaluations", "train_mse", "test_mse",
          "test_accuracy", "training_seconds", "hidden_used", "enabled_connections")
METHOD_IDS = ("classic_neat", "independent", "waltz_fixed_5", "waltz_accelerating")
EXCHANGE_METHOD_IDS = ("classic_neat", "independent", "waltz_fixed_5", "waltz_stationary", "ring_migration")


def _run_trial(config, train, test, generations, seed, method, waltzers, candidates, preview):
    """Train one method/seed in isolation; only the parent writes result files."""
    budget = generations * 4 * config.pop_size
    evaluations = 0

    def fitness(genome, model_config):
        nonlocal evaluations
        evaluations += 1
        network = neat.nn.FeedForwardNetwork.create(genome, model_config)
        return -classification_metrics(network, train)[0]

    print(f"Seed {seed} | {METHODS[method][0]} | {budget:,} training evaluations...", flush=True)
    start = perf_counter()
    if method == "classic_neat":
        genome, model_config = run_classic_neat(config, fitness, generations, seed)
    elif method == "ring_migration":
        genome = run_ring_migration(config, fitness, generations, seed=seed, interval=5)
        model_config = config
    else:
        options = {"tempo_schedule": (15, 10, 5)} if method == "waltz_accelerating" else {"tempo": 5}
        genome, _ = run_waltz(config, fitness, generations, seed=seed,
                              waltzers=0 if method == "independent" else waltzers,
                              travel=method != "waltz_stationary",
                              candidates_per_waltzer=candidates, **options)
        model_config = config
    seconds = perf_counter() - start
    if evaluations != budget:
        raise RuntimeError(f"Expected {budget} evaluations, observed {evaluations}")
    network = neat.nn.FeedForwardNetwork.create(genome, model_config)
    test_mse, test_accuracy = classification_metrics(network, test)
    row = dict(generations=generations, seed=seed, method=method, genome_evaluations=evaluations,
               train_mse=-genome.fitness, test_mse=test_mse, test_accuracy=test_accuracy,
               training_seconds=seconds,
               hidden_used=sum(key not in model_config.genome_config.output_keys for key, *_ in network.node_evals),
               enabled_connections=sum(gene.enabled for gene in genome.connections.values()))
    model = None
    if preview:
        _, train_accuracy = classification_metrics(network, train)
        model = dict(name=METHODS[method][0], network=network, train_mse=-genome.fitness,
                     test_mse=test_mse, train_accuracy=train_accuracy, test_accuracy=test_accuracy,
                     hidden_stored=sum(key not in model_config.genome_config.output_keys for key in genome.nodes),
                     hidden_used=row["hidden_used"], enabled=row["enabled_connections"], links=len(genome.connections))
    return row, model


def _trial_results(jobs, workers):
    """Spawn independent interpreters so Python RNG and NEAT state are separate."""
    if workers == 1:
        for job in jobs:
            yield _run_trial(*job)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(_run_trial, *job) for job in jobs]
        try:
            for future in as_completed(futures):
                yield future.result()
        finally:
            for future in futures:
                future.cancel()


def run_comparison(config, train, test, generations, seeds, output, metadata, *, waltzers=4, candidates=1,
                   preview=False, workers=1, methods=METHOD_IDS):
    """Save each completed trial; optional processes parallelize independent runs."""
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    seeds = tuple(seeds)
    methods = tuple(methods)
    if not methods or len(methods) != len(set(methods)) or any(method not in set(METHOD_IDS + EXCHANGE_METHOD_IDS) for method in methods):
        raise ValueError("Use a nonempty sequence of distinct supported methods")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Use a nonempty sequence of distinct evolution seeds")
    workers = min(workers, len(seeds) * len(methods))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = Path(f"{output}.csv"), Path(f"{output}.json")
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("Output CSV/JSON already exists; use a new --output prefix")
    if preview and metadata.get("task", "").lower() not in ("circle", "spiral"):
        raise ValueError("Boundary previews require Circle or Spiral")
    budget = generations * 4 * config.pop_size
    metadata = dict(metadata, created_utc=datetime.now(timezone.utc).isoformat(),
                    generations=generations, evolution_seeds=seeds, population_total=4 * config.pop_size,
                    island_count=4, island_population=config.pop_size, waltzers=waltzers,
                    candidates_per_waltzer=candidates, initial_hidden=config.genome_config.num_hidden,
                    initialization=INITIALIZATION,
                    weight_mutate_power=config.genome_config.weight_mutate_power,
                    bias_mutate_power=config.genome_config.bias_mutate_power,
                    fixed_tempo=5, tempo_schedule=[15, 10, 5], training_evaluations_per_run=budget,
                    selection="Highest training fitness seen (negative training MSE)",
                    training_point_count=len(train), test_point_count=len(test),
                    training_point_predictions_per_run=budget * len(train),
                    training_examples=train, test_examples=test, python_version=platform.python_version(),
                    neat_python_version=version("neat-python"), platform=platform.platform(),
                    bootstrap_resamples=10000, bootstrap_seed=0,
                    workers=workers, multiprocessing_start_method="spawn" if workers > 1 else None,
                    result_order="completion" if workers > 1 else "seed_then_method",
                    primary_method="waltz_fixed_5",
                    primary_baselines=metadata.get("primary_baselines", ["classic_neat", "independent"]),
                    secondary_method="waltz_accelerating" if "waltz_accelerating" in methods else None,
                    boundary_previews_per_seed=preview,
                    methods=methods,
                    neat_settings=NEAT_SETTINGS
                        .replace("num_hidden = 0", f"num_hidden = {config.genome_config.num_hidden}")
                        .replace("weight_mutate_power = 0.5", f"weight_mutate_power = {config.genome_config.weight_mutate_power}")
                        .replace("bias_mutate_power = 0.5", f"bias_mutate_power = {config.genome_config.bias_mutate_power}"))
    source_dir = Path(__file__).parent
    metadata["source_sha256"] = {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                                 for name in ("compare_models.py", "plot_results.py", "neat_genome.py", "neat_waltz.py", "test_xor.py",
                                              "diagnose_circle.py", "test_circle.py", "test_spiral.py", "confirm_waltz.py",
                                              "neat_migration.py", "confirm_exchange.py", "preview_spiral.py")}
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    jobs = [(config, train, test, generations, seed, method, waltzers, candidates, preview)
            for seed in seeds for method in methods]
    print(f"Running {len(jobs)} trials with {workers} worker(s). "
          f"Training/test points: {len(train)}/{len(test)}.", flush=True)
    preview_paths = []
    pending_models = {}
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        stream.flush()
        for completed, (row, model) in enumerate(_trial_results(jobs, workers), start=1):
            writer.writerow(row)
            stream.flush()
            seed, method = row["seed"], row["method"]
            print(f"Completed {completed}/{len(jobs)} | Seed {seed} | {METHODS[method][0]} | "
                  f"test {row['test_accuracy']:.2%} | train MSE {row['train_mse']:.6f} | "
                  f"{row['training_seconds']:.1f}s", flush=True)
            if preview:
                pending_models.setdefault(seed, {})[method] = (row, model)
                if len(pending_models[seed]) == len(methods):
                    trials = [pending_models[seed][method] for method in methods]
                    path = Path(f"{output}_seed{seed}_boundaries.png")
                    if metadata["task"].lower() == "circle":
                        save_diagnostic(path, [model for _, model in trials], test, seed, generations,
                                        metadata.get("test_data_seed"), initial_hidden=config.genome_config.num_hidden,
                                        training_points=len(train), mutation_power=config.genome_config.weight_mutate_power)
                    else:
                        from preview_spiral import save_previews
                        save_previews(path.with_suffix(""), trials, metadata, comparison=True)
                    preview_paths.append(path)
                    del pending_models[seed]
                    print(f"Boundary preview saved to {path}", flush=True)
    return [*preview_paths, *save_plots(csv_path)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("circle", "spiral", "xor"), default="circle")
    parser.add_argument("--generations", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs=2, default=(35, 54), metavar=("FIRST", "LAST"))
    parser.add_argument("--test-seed", type=int, default=13)
    parser.add_argument("--train-size", type=int,
                        help="Balanced Circle/Spiral training points, even and positive (default: 128)")
    parser.add_argument("--mutation-power", type=float, default=0.5,
                        help="Gaussian mutation standard deviation for weights and biases (default: 0.5)")
    parser.add_argument("--hidden", type=int, default=4)
    parser.add_argument("--waltzers", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--stage", choices=("exploratory", "confirmation"), default="confirmation")
    parser.add_argument("--workers", type=int, default=1, help="Independent training processes (default: 1)")
    parser.add_argument("--no-preview", action="store_true", help="Skip automatic per-seed Circle boundary previews")
    parser.add_argument("--output", type=Path, help="Output prefix (default: results/TASK_confirmation)")
    args = parser.parse_args()
    if args.generations < 1 or args.seeds[0] > args.seeds[1] or args.hidden < 0:
        parser.error("Use positive generations, nonnegative hidden nodes and FIRST <= LAST")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not 1 <= args.waltzers <= 4 or not 1 <= args.candidates < 30:
        parser.error("Use 1–4 waltzers and 1–29 candidates")
    if not isfinite(args.mutation_power) or args.mutation_power < 0:
        parser.error("--mutation-power must be finite and nonnegative")
    if args.train_size is not None and (args.task == "xor" or args.train_size < 2 or args.train_size % 2):
        parser.error("--train-size requires Circle/Spiral and a positive even number of points")
    if args.task == "xor":
        train = test = XOR
    else:
        generator = circle_examples if args.task == "circle" else spiral_examples
        train_size = 128 if args.train_size is None else args.train_size
        train, test = generator(train_size // 2, seed=10), generator(128, seed=args.test_seed)
        if {inputs for inputs, _ in train} & {inputs for inputs, _ in test}:
            parser.error("Training/test inputs overlap; choose another --test-seed")
    output = args.output or Path(f"results/{args.task}_{args.stage}")
    metadata = dict(task=args.task.title(), stage=args.stage,
                    train_data_seed=None if args.task == "xor" else 10,
                    test_data_seed=None if args.task == "xor" else args.test_seed,
                    metric="Truth-table accuracy" if args.task == "xor" else "Held-out test accuracy",
                    source="Direct experiment output; full-precision accuracies")
    try:
        paths = run_comparison(create_config(args.hidden, mutation_power=args.mutation_power),
                               train, test, args.generations,
                               range(args.seeds[0], args.seeds[1] + 1), output, metadata,
                               waltzers=args.waltzers, candidates=args.candidates,
                               preview=args.task == "circle" and not args.no_preview, workers=args.workers)
    except FileExistsError as error:
        parser.error(str(error))
    for path in (Path(f"{output}.csv"), Path(f"{output}.json"), *paths):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
