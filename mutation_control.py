"""Mutation-only ablation paired with the frozen accelerating tempo pilot."""

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
from statistics import median
from time import perf_counter

import neat
import numpy as np
from matplotlib.figure import Figure

import pilot_tempo as pilot


def trial(task, dataset, seed, *, smoke=False):
    generator = {"circle": pilot.circle_examples, "spiral": pilot.spiral_examples}[task]
    train_seed, test_seed = pilot.DATA_SEEDS[dataset]
    train = generator(8 if smoke else 64, train_seed)
    test = generator(16 if smoke else 256, test_seed)
    if {x for x, _ in train} & {x for x, _ in test}:
        raise RuntimeError("Training and test inputs overlap")
    config = pilot.create_config(4, mutation_power=0.1)
    evaluations, children, history, visits = 0, [], [], []

    def fitness(genome, cfg):
        nonlocal evaluations
        evaluations += 1
        return -pilot.classification_metrics(neat.nn.FeedForwardNetwork.create(genome, cfg), train)[0]

    started = perf_counter()
    best, final = pilot.run_waltz(
        config, fitness, pilot.GENERATIONS, seed=seed, crossover=False,
        **pilot.visit_options("accelerating", pilot.GENERATIONS),
        on_candidate=lambda **e: children.append(e),
        on_generation=lambda **e: history.append(e), on_visit=lambda **e: visits.append(e),
    )
    seconds = perf_counter() - started
    if (evaluations != pilot.GENERATIONS * 120 or len(children) != pilot.GENERATIONS - 1
            or any(e["local_parent"] is not None for e in children)):
        raise RuntimeError("Mutation-only budget or parent mismatch")
    best_test = pilot.classification_metrics(neat.nn.FeedForwardNetwork.create(best, config), test)
    final_test = pilot.classification_metrics(neat.nn.FeedForwardNetwork.create(final, config), test)
    accepted = [e for e in children if e["accepted"]]
    row = dict(task=task, dataset=dataset, seed=seed, method="mutation_only", generations=pilot.GENERATIONS,
               genome_evaluations=evaluations, training_predictions=evaluations * len(train),
               candidate_evaluations=len(children), training_seconds=seconds,
               train_mse=-best.fitness, test_mse=best_test[0], test_accuracy=best_test[1],
               final_waltzer_train_mse=-final.fitness, final_waltzer_test_mse=final_test[0],
               final_waltzer_test_accuracy=final_test[1],
               moves=sum(a["island"] != b["island"] for a, b in zip(visits, visits[1:])),
               visits=len(visits), accepted=len(accepted), acceptance_rate=len(accepted) / len(children),
               accepted_gain=sum(e["fitness_gain"] for e in accepted))
    return dict(metrics=row, candidate_history=children, generation_history=history, visits=visits)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(output, controls, references):
    """Positive paired gains always favor crossover plus mutation."""
    pairs = []
    for control in controls:
        m = control["metrics"]
        key = tuple(m[k] for k in ("task", "dataset", "seed"))
        c = references[key]["metrics"]
        row = dict(zip(("task", "dataset", "seed"), key))
        for field in ("test_accuracy", "final_waltzer_test_accuracy"):
            row[field + "_gain_pp"] = (c[field] - m[field]) * 100
        for field in ("test_mse", "final_waltzer_test_mse"):
            row[field + "_reduction"] = m[field] - c[field]
        pairs.append(row)
    write_csv(output / "paired_differences.csv", pairs)
    summary = []
    for task in ("circle", "spiral"):
        for dataset in (0, 1):
            local = [p for p in pairs if p["task"] == task and p["dataset"] == dataset]
            if not local:
                continue
            for metric in list(local[0])[3:]:
                gains = [r[metric] for r in local]
                summary.append(dict(task=task, dataset=dataset, metric=metric, n=len(gains),
                                    median_gain=median(gains), wins=sum(g > 1e-12 for g in gains),
                                    ties=sum(abs(g) <= 1e-12 for g in gains),
                                    losses=sum(g < -1e-12 for g in gains)))
    write_csv(output / "paired_summary.csv", summary)
    write_csv(output / "scores.csv", [r["metrics"] for r in controls] + [r["metrics"] for r in references.values()])
    for lineage in (False, True):
        figure = Figure(figsize=(12, 8), layout="constrained")
        axes = figure.subplots(2, 2)
        for task_index, task in enumerate(("circle", "spiral")):
            for dataset in (0, 1):
                ax = axes[task_index, dataset]
                for label, source in (("Crossover + mutation", references.values()), ("Mutation only", controls)):
                    traces = [r["generation_history"] for r in source
                              if r["metrics"]["task"] == task and r["metrics"]["dataset"] == dataset]
                    if not traces:
                        continue
                    values = [[-e["lineages"][0]["fitness"] if lineage else -e["best_fitness"] for e in h] for h in traces]
                    ax.plot([e["genome_evaluations"] for e in traces[0]], np.median(values, axis=0), label=label)
                ax.set(title=f"{task.title()} | data {dataset}", xlabel="Genome evaluations", ylabel="Median training MSE")
                ax.legend(fontsize=8)
                for boundary in (120, 240):
                    ax.axvline((boundary + 1) * 120, color="gray", linestyle="--", linewidth=0.7)
        figure.suptitle("Final-lineage training progress" if lineage else "Best-global training progress")
        figure.savefig(output / ("lineage_progress.png" if lineage else "training_progress.png"), dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=Path("generated/tempo_pilot_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("generated/mutation_control_v1"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--smoke-test", action="store_true", help="Two small-data checks, not compared to full-data references")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Use positive workers")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    original = json.loads((args.reference_dir / "protocol.json").read_text())
    for name, digest in original["source_sha256"].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != digest:
            parser.error(f"Reference source changed: {name}; do not reuse the old scores silently")
    if (original["stage"] != "exploratory_pilot" or original["python_version"] != platform.python_version()
            or original["neat_python_version"] != version("neat-python")):
        parser.error("Reference protocol or runtime is incompatible")
    jobs = [(t, d, s) for t in ("circle", "spiral")
            for d in ((0,) if args.smoke_test else range(2))
            for s in (pilot.EVOLUTION_SEEDS[:1] if args.smoke_test else pilot.EVOLUTION_SEEDS)]
    references, reference_hashes = {}, {}
    if not args.smoke_test:
        for job in jobs:
            name = pilot.run_name((*job, "accelerating")) + ".json"
            raw = (args.reference_dir / name).read_bytes()
            result = json.loads(raw)
            if tuple(result["metrics"][k] for k in ("task", "dataset", "seed")) != job:
                raise RuntimeError("Reference identity mismatch")
            references[job] = result
            reference_hashes[name] = hashlib.sha256(raw).hexdigest()
    protocol = dict(stage="software_smoke" if args.smoke_test else "exploratory_mutation_ablation",
                    parent_protocol_sha256=hashlib.sha256((args.reference_dir / "protocol.json").read_bytes()).hexdigest(),
                    settings=original, jobs=jobs, crossover=False, smoke=args.smoke_test,
                    only_change="Clone and mutate the retained waltzer, instead of crossover then mutation",
                    reference_sha256=reference_hashes, analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    limitations="Same seed does not mean identical random draws after crossover is removed; exploratory reused data, not fresh confirmation")
    protocol = json.loads(json.dumps(protocol))
    output = args.output_dir
    metadata = output / "protocol.json"
    if args.resume:
        if not metadata.exists() or json.loads(metadata.read_text()) != protocol:
            parser.error("Resume requires identical sources, references and protocol")
    else:
        if output.exists() and any(output.iterdir()):
            parser.error("Use a fresh output directory or --resume")
        output.mkdir(parents=True, exist_ok=True)
        with metadata.open("x") as stream:
            json.dump(protocol, stream, indent=2)
    controls, remaining = [], []
    for job in jobs:
        path = output / (pilot.run_name((*job, "mutation_only")) + ".json")
        if path.exists():
            controls.append(json.loads(path.read_text()))
        else:
            remaining.append(job)
    print(f"Mutation control: {len(jobs)} planned runs; {len(controls)} already saved; {args.workers} workers.", flush=True)
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        futures = {pool.submit(trial, *job, smoke=args.smoke_test): job for job in remaining}
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            path = output / (pilot.run_name((*job, "mutation_only")) + ".json")
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(result, separators=(",", ":")))
            temporary.replace(path)
            controls.append(result)
            print(f"{len(controls)}/{len(jobs)} | {path.stem} | {result['metrics']['training_seconds']:.1f}s", flush=True)
    if not args.smoke_test:
        compare(output, controls, references)
    (output / "timing.json").write_text(json.dumps(dict(wall_seconds=perf_counter() - started,
                                                       completed_now=len(remaining), total_completed=len(controls), workers=args.workers), indent=2))
    print(f"Saved {len(controls)} mutation-only runs to {output}.", flush=True)


if __name__ == "__main__":
    main()
