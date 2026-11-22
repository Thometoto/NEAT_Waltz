"""Fresh local/remote champion crossover, without a retained breeding lineage."""

import argparse
import hashlib
import json
import os
import platform
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import copy, deepcopy
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path
from statistics import median
from time import perf_counter

import neat
import numpy as np
from matplotlib.figure import Figure

import pilot_tempo as pilot
from mutation_control import write_csv
from neat_waltz import (_check_population_sizes, _create_populations, _evaluate_champion,
                        _generation_event, _genome_event, _insert_waltzer_candidate, _reproduce)


def run_nonpersistent(config, fitness, generations, *, seed, tempo_schedule):
    """Return best global and diagnostic stream archive; neither breeds externally.

    Every round inserts one child into the scheduled host. The remote parent is
    the current champion of (host - 1) % 4, not an externally retained genome.
    Both parents are snapshotted before any island reproduces in that round.
    The stream archive starts with island 0's initial champion, matching Waltz's
    initial candidate for output selection, but is never used as a parent.
    """
    if type(generations) is not int or generations < 1:
        raise ValueError("Use positive integer generations")
    tempos = tuple(tempo_schedule)
    if not tempos or any(type(t) is not int or t < 1 for t in tempos):
        raise ValueError("Use a nonempty positive integer tempo schedule")
    config = copy(config)
    gc = config.genome_config
    original_indexer, original_tracker = gc.node_indexer, gc.innovation_tracker
    state = random.getstate()
    try:
        random.seed(seed)
        config.seed = None
        gc.node_indexer = None
        populations = _create_populations(config, 4)
        best = stream_best = pending = None
        children, history, visits = [], [], []
        evaluations = 0
        visit_index, first, host = 0, 1, 0
        tempo = end = tempos[0]
        visit_stream_before = None
        for generation in range(generations):
            _check_population_sizes(populations, config.pop_size)
            champions = [_evaluate_champion(p, fitness) for p in populations]
            evaluations += sum(len(p.population) for p in populations)
            champion = max(champions, key=lambda g: g.fitness)
            if best is None or champion.fitness > best.fitness:
                best = deepcopy(champion)
            if generation == 0:
                stream_best = deepcopy(champions[0])
                visit_stream_before = stream_best.fitness
            else:
                child, parents = pending
                new_record = child.fitness > stream_best.fitness
                children.append(dict(generation=generation, created_generation=generation - 1,
                                     host=host, tempo=tempo, **parents,
                                     candidate=_genome_event(child, host),
                                     beats_both=child.fitness > max(parents['remote_parent']['fitness'],
                                                                   parents['local_parent']['fitness']),
                                     new_stream_record=new_record))
                if new_record:
                    stream_best = deepcopy(child)
            event = _generation_event(generation, populations, best, [], [], [], evaluations, len(children))
            event.update(host=host, stream_best_fitness=stream_best.fitness)
            history.append(event)
            final = generation + 1 == generations
            if generation > 0 and (generation == end or final):
                visits.append(dict(island=host, first_generation=first, last_generation=generation,
                                   tempo=tempo, stream_before=visit_stream_before,
                                   stream_after=stream_best.fitness))
                visit_stream_before = stream_best.fitness
            if final:
                break
            if generation == end:
                visit_index += 1
                tempo = tempos[min(visit_index // 4, len(tempos) - 1)]
                first, end = generation + 1, generation + tempo
            host = visit_index % 4
            donor = (host - 1) % 4
            # These are fresh champions, chosen before any new child is created.
            remote, local = deepcopy(champions[donor]), deepcopy(champions[host])
            parents = dict(remote_parent=_genome_event(remote, donor), local_parent=_genome_event(local, host))
            for index, population in enumerate(populations):
                offspring = _reproduce(population, generation)
                if index == host:
                    child = _insert_waltzer_candidate(population, offspring, remote, local)
                    pending = child, parents
                population.population = offspring
                population.species.speciate(config, offspring, generation)
        if evaluations != generations * 4 * config.pop_size or len(children) != generations - 1:
            raise RuntimeError("Nonpersistent evaluation budget mismatch")
        return dict(best=best, stream_best=stream_best, candidate_history=children,
                    generation_history=history, visits=visits)
    finally:
        gc.node_indexer, gc.innovation_tracker = original_indexer, original_tracker
        random.setstate(state)


def trial(task, dataset, seed, *, smoke=False):
    generator = {"circle": pilot.circle_examples, "spiral": pilot.spiral_examples}[task]
    train_seed, test_seed = pilot.DATA_SEEDS[dataset]
    train = generator(8 if smoke else 64, train_seed)
    test = generator(16 if smoke else 256, test_seed)
    if {x for x, _ in train} & {x for x, _ in test}:
        raise RuntimeError("Training and test inputs overlap")
    config = pilot.create_config(4, mutation_power=0.1)
    calls = 0

    def fitness(genome, cfg):
        nonlocal calls
        calls += 1
        return -pilot.classification_metrics(neat.nn.FeedForwardNetwork.create(genome, cfg), train)[0]

    schedule = pilot.visit_options("accelerating", pilot.GENERATIONS)["tempo_schedule"]
    started = perf_counter()
    result = run_nonpersistent(config, fitness, pilot.GENERATIONS, seed=seed, tempo_schedule=schedule)
    seconds = perf_counter() - started
    best, stream = result.pop("best"), result.pop("stream_best")
    bm = pilot.classification_metrics(neat.nn.FeedForwardNetwork.create(best, config), test)
    sm = pilot.classification_metrics(neat.nn.FeedForwardNetwork.create(stream, config), test)
    if calls != pilot.GENERATIONS * 120:
        raise RuntimeError("Actual fitness-call budget mismatch")
    result["metrics"] = dict(task=task, dataset=dataset, seed=seed, method="nonpersistent",
                             generations=pilot.GENERATIONS, genome_evaluations=calls,
                             training_predictions=calls * len(train), candidate_evaluations=len(result["candidate_history"]),
                             training_seconds=seconds, train_mse=-best.fitness, test_mse=bm[0], test_accuracy=bm[1],
                             stream_train_mse=-stream.fitness, stream_test_mse=sm[0], stream_test_accuracy=sm[1],
                             moves=sum(a["island"] != b["island"] for a, b in zip(result["visits"], result["visits"][1:])))
    return result


def compare(output, controls, references):
    pairs, scores = [], []
    for control in controls:
        r = control["metrics"]
        key = tuple(r[k] for k in ("task", "dataset", "seed"))
        w = references[key]["metrics"]
        pairs.append(dict(zip(("task", "dataset", "seed"), key),
                          best_accuracy_gain_pp=(w["test_accuracy"] - r["test_accuracy"]) * 100,
                          best_mse_reduction=r["test_mse"] - w["test_mse"],
                          stream_accuracy_gain_pp=(w["final_waltzer_test_accuracy"] - r["stream_test_accuracy"]) * 100,
                          stream_mse_reduction=r["stream_test_mse"] - w["final_waltzer_test_mse"]))
        scores.append(r)
        scores.append(dict((k, w[k]) for k in ("task", "dataset", "seed", "method", "generations", "genome_evaluations",
                                              "training_predictions", "candidate_evaluations", "training_seconds",
                                              "train_mse", "test_mse", "test_accuracy", "moves")) |
                      dict(stream_train_mse=w["final_waltzer_train_mse"], stream_test_mse=w["final_waltzer_test_mse"],
                           stream_test_accuracy=w["final_waltzer_test_accuracy"]))
    write_csv(output / "scores.csv", scores)
    write_csv(output / "paired_differences.csv", pairs)
    summary = []
    for task in ("circle", "spiral"):
        for dataset in (0, 1):
            local = [r for r in pairs if r["task"] == task and r["dataset"] == dataset]
            for metric in list(local[0])[3:]:
                gains = [r[metric] for r in local]
                summary.append(dict(task=task, dataset=dataset, metric=metric, n=len(gains), median_gain=median(gains),
                                    wins=sum(g > 1e-12 for g in gains), ties=sum(abs(g) <= 1e-12 for g in gains),
                                    losses=sum(g < -1e-12 for g in gains)))
    write_csv(output / "paired_summary.csv", summary)
    for stream in (False, True):
        figure = Figure(figsize=(12, 8), layout="constrained")
        axes = figure.subplots(2, 2)
        for i, task in enumerate(("circle", "spiral")):
            for dataset in (0, 1):
                ax = axes[i, dataset]
                for label, source in (("Persistent Waltz", references.values()), ("Fresh remote champion", controls)):
                    runs = [r for r in source if r["metrics"]["task"] == task and r["metrics"]["dataset"] == dataset]
                    curves = [[-e["best_fitness"] if not stream else -(
                        e["lineages"][0]["fitness"] if e["lineages"] else e["stream_best_fitness"])
                               for e in r["generation_history"]] for r in runs]
                    ax.plot([e["genome_evaluations"] for e in runs[0]["generation_history"]],
                            np.median(curves, axis=0), label=label)
                ax.set(title=f"{task.title()} | data {dataset}", xlabel="Genome evaluations", ylabel="Median training MSE")
                ax.legend(fontsize=8)
        figure.suptitle("Injected-stream diagnostic archive (not a control breeding parent)" if stream else "Best-global archive")
        figure.savefig(output / ("stream_progress.png" if stream else "training_progress.png"), dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=Path("generated/tempo_pilot_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("generated/nonpersistent_control_v1"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Use positive workers")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    original = json.loads((args.reference_dir / "protocol.json").read_text())
    for name, digest in original["source_sha256"].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != digest:
            parser.error(f"Reference source changed: {name}")
    if (original["stage"] != "exploratory_pilot" or original["python_version"] != platform.python_version()
            or original["neat_python_version"] != version("neat-python")):
        parser.error("Incompatible reference or runtime")
    jobs = [(t, d, s) for t in ("circle", "spiral") for d in ((0,) if args.smoke_test else range(2))
            for s in (pilot.EVOLUTION_SEEDS[:1] if args.smoke_test else pilot.EVOLUTION_SEEDS)]
    references, hashes = {}, {}
    if not args.smoke_test:
        for job in jobs:
            name = pilot.run_name((*job, "accelerating")) + ".json"
            raw = (args.reference_dir / name).read_bytes()
            references[job] = json.loads(raw)
            if tuple(references[job]["metrics"][k] for k in ("task", "dataset", "seed")) != job:
                raise RuntimeError("Reference identity mismatch")
            hashes[name] = hashlib.sha256(raw).hexdigest()
    protocol = dict(stage="software_smoke" if args.smoke_test else "exploratory_nonpersistent_ablation",
                    settings=original, jobs=jobs, smoke=args.smoke_test, reference_sha256=hashes,
                    parent_protocol_sha256=hashlib.sha256((args.reference_dir / "protocol.json").read_bytes()).hexdigest(),
                    source_sha256={n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                                   for n in ("nonpersistent_control.py", "mutation_control.py")},
                    donor_policy="Fresh champion of (host - 1) % 4, selected before any island reproduces",
                    stream_archive="Initial island-0 champion plus injected children; diagnostic only, never a parent",
                    limitations="Changes the first-parent policy and removes external breeding memory; does not isolate a latent memory variable")
    protocol = json.loads(json.dumps(protocol))
    output = args.output_dir
    metadata = output / "protocol.json"
    if args.resume:
        if not metadata.exists() or json.loads(metadata.read_text()) != protocol:
            parser.error("Resume requires identical sources, references and settings")
    else:
        if output.exists() and any(output.iterdir()):
            parser.error("Use a fresh output directory or --resume")
        output.mkdir(parents=True, exist_ok=True)
        with metadata.open("x") as f:
            json.dump(protocol, f, indent=2)
    controls, remaining = [], []
    for job in jobs:
        path = output / (pilot.run_name((*job, "nonpersistent")) + ".json")
        if path.exists():
            result = json.loads(path.read_text())
            if tuple(result["metrics"][k] for k in ("task", "dataset", "seed")) != job:
                raise RuntimeError("Saved control identity mismatch")
            controls.append(result)
        else:
            remaining.append(job)
    print(f"Nonpersistent control: {len(jobs)} planned, {len(controls)} saved; {args.workers} workers.", flush=True)
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        futures = {pool.submit(trial, *job, smoke=args.smoke_test): job for job in remaining}
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            path = output / (pilot.run_name((*job, "nonpersistent")) + ".json")
            temp = path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(result, separators=(",", ":")))
            temp.replace(path)
            controls.append(result)
            print(f"{len(controls)}/{len(jobs)} | {path.stem} | {result['metrics']['training_seconds']:.1f}s", flush=True)
    if not args.smoke_test:
        compare(output, controls, references)
    (output / "timing.json").write_text(json.dumps(dict(wall_seconds=perf_counter() - started,
                                                       completed_now=len(remaining), total_completed=len(controls), workers=args.workers), indent=2))
    print(f"Saved {len(controls)} runs to {output}.", flush=True)


if __name__ == "__main__":
    main()
