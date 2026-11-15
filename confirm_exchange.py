"""Confirm Waltz travel against stationary lineages and champion ring migration."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3, help="Simultaneous training processes (default: 3)")
    parser.add_argument("--task", choices=("both", "circle", "spiral"), default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("results/waltz_exchange_confirmation"))
    parser.add_argument("--no-preview", action="store_true", help="Skip automatic per-seed boundary images")
    parser.add_argument("--smoke-test", action="store_true", help="Short software check; not scientific evidence")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.smoke_test and args.output_dir == Path("results/waltz_exchange_confirmation"):
        args.output_dir = Path("results/waltz_exchange_smoke")

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"

    from compare_models import EXCHANGE_METHOD_IDS, run_comparison
    from diagnose_circle import create_config
    from test_circle import circle_examples
    from test_spiral import spiral_examples

    tasks = ("circle", "spiral") if args.task == "both" else (args.task,)
    outputs = [args.output_dir.resolve() / task for task in tasks]
    # Check all requested tasks before training; preserve historical confirmations.
    for output in outputs:
        if any(Path(f"{output}{suffix}").exists() for suffix in (".csv", ".json")):
            parser.error(f"Existing results at {output}; use a new --output-dir or select an unstarted --task")

    generations = 6 if args.smoke_test else 2000
    seeds = range(5, 7) if args.smoke_test else range(75, 95)
    train_seed, test_seed = (10, 11) if args.smoke_test else (15, 16)
    train_size, test_size = (16, 16) if args.smoke_test else (512, 256)
    for task, output in zip(tasks, outputs):
        generator = circle_examples if task == "circle" else spiral_examples
        train = generator(train_size // 2, seed=train_seed)
        test = generator(test_size // 2, seed=test_seed)
        if {inputs for inputs, _ in train} & {inputs for inputs, _ in test}:
            raise ValueError("Training and test inputs overlap")
        metadata = dict(
            task=task.title(), stage="Software smoke test" if args.smoke_test else "Exchange confirmation",
            protocol="waltz_exchange_v1", train_data_seed=train_seed, test_data_seed=test_seed,
            metric="Held-out test accuracy", source="Direct experiment output; full-precision accuracies",
            primary_baselines=["waltz_stationary", "ring_migration"],
            secondary_baselines=["classic_neat", "independent"],
            ring_migration=dict(interval=5, copies_per_island=1, direction="i -> (i+1) mod 4",
                                first_evaluated_generation=5,
                                replacement="Unevaluated offspring first, then lowest-fitness elite",
                                selection="Current source-island champion; simultaneous donor snapshots",
                                mutation=False, persistent_lineage=False),
            stationary_control="Four retained lineages; crossover and mutation unchanged; no host changes",
            dataset_scope="One fixed fresh training/test pair per task, shared by every method and evolution seed",
            inference_scope="Conditional on fixed datasets; pointwise bootstrap intervals, no multiplicity correction",
            tuning="Shared hyperparameters, no separate baseline tuning; no tuning on confirmation results",
            migration_event_count=(generations - 1) // 5,
            migrant_evaluations=4 * ((generations - 1) // 5),
            waltzer_candidate_evaluations=4 * (generations - 1),
        )
        print(f"Starting {task.title()} | {len(seeds)} seeds | {len(EXCHANGE_METHOD_IDS)} methods | "
              f"{generations * 120:,} training evaluations per run | {args.workers} worker(s).", flush=True)
        paths = run_comparison(create_config(4, mutation_power=0.1), train, test, generations,
                               seeds, output, metadata, methods=EXCHANGE_METHOD_IDS,
                               waltzers=4, candidates=1, workers=args.workers, preview=not args.no_preview)
        for path in (Path(f"{output}.csv"), Path(f"{output}.json"), *paths):
            print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
