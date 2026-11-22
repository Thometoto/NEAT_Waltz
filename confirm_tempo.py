"""Last prespecified Waltz comparison: Circle, Spiral and domain-specialized Sine."""

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

from diagnose_circle import classification_metrics, create_config, save_diagnostic
from neat_genome import INITIALIZATION
from neat_waltz import run_waltz
from neat_waltz_domains import region_errors, run_domains
from test_circle import circle_examples
from test_sine import shade_regions, sine, sine_config, sine_regions
from test_spiral import spiral_examples
from test_xor import NEAT_SETTINGS, run_classic_neat
from waltz_tempo import balanced_schedule


METHODS = dict(classic="Classic NEAT", independent="Independent islands",
               fixed5="Waltz, fixed 5", fixed15="Waltz, fixed 15",
               accelerating="Waltz, 15 → 10 → 5 (three phases)", stationary="Stationary lineages")
FIELDS = ("task", "seed", "method", "generations", "genome_evaluations", "training_predictions",
          "candidate_evaluations", "train_mse", "test_mse", "test_accuracy",
          "final_waltzer_train_mse", "final_waltzer_test_mse", "final_waltzer_test_accuracy", "training_seconds")


def make_data(task, smoke=False):
    """Keep earlier data distributions; fresh fixed data, shared across methods/seeds."""
    if task == "sine":
        return sine_regions(4 if smoke else 128, 22), sine_regions(8 if smoke else 256, 23)
    generator = circle_examples if task == "circle" else spiral_examples
    return generator(8 if smoke else 256, seed=17), generator(16 if smoke else 128, seed=18)


def run_trial(task, train, test, generations, seed, method):
    """Training uses no test data. Store the best training genome and final lineage separately."""
    schedule, _ = balanced_schedule(generations)
    options = dict(tempo=15 if method == "fixed15" else 5)
    if method in ("accelerating", "stationary"):
        options["tempo_schedule"] = schedule
    is_sine = task == "sine"
    waltzers = 1 if is_sine else 4
    config = sine_config() if is_sine else create_config(4, mutation_power=0.1)
    visits, evaluations = [], 0
    candidate_history, generation_history = [], []
    observers = dict(on_candidate=lambda **event: candidate_history.append(event),
                     on_generation=lambda **event: generation_history.append(event))
    print(f"{task.title()} | seed {seed} | {METHODS[method]} | {generations * 120:,} evaluations...", flush=True)
    started = perf_counter()
    if is_sine:
        mode = dict(classic="classic", independent="independent", stationary="stationary").get(method, "waltz")
        result = run_domains(config, train, generations, mode=mode, seed=seed, waltzers=waltzers,
                             **options, **observers)
        best = result["best"]
        final = max(result["lineages"], key=lambda g: g.fitness, default=None)
        evaluations = result["genome_evaluations"]
        for event in result["visits"]:
            visits.append(dict(event, before=-sum(event["region_mse_before"]) / 4,
                               after=-sum(event["region_mse_after"]) / 4))
        history = result["history"]
        train_count = sum(map(len, train))
        candidate_evaluations = result["candidate_evaluations"]
    else:
        def fitness(genome, cfg):
            nonlocal evaluations
            evaluations += 1
            return -classification_metrics(neat.nn.FeedForwardNetwork.create(genome, cfg), train)[0]

        if method == "classic":
            best, config = run_classic_neat(config, fitness, generations, seed,
                                           on_generation=observers["on_generation"])
            final = None
        else:
            best, final = run_waltz(config, fitness, generations, seed=seed,
                                   waltzers=0 if method == "independent" else waltzers,
                                   travel=method != "stationary", on_visit=lambda **event: visits.append(event),
                                   **options, **observers)
        history = []
        train_count = len(train)
        candidate_evaluations = len(candidate_history)
    seconds = perf_counter() - started
    if evaluations != generations * 120:
        raise RuntimeError("Training evaluation budget mismatch")

    def metrics(genome):
        if genome is None:
            return "", ""
        network = neat.nn.FeedForwardNetwork.create(genome, config)
        return (sum(region_errors(network, test)) / 4, "") if is_sine else classification_metrics(network, test)

    test_mse, test_accuracy = metrics(best)
    final_mse, final_accuracy = metrics(final)
    row = dict(task=task, seed=seed, method=method, generations=generations, genome_evaluations=evaluations,
               training_predictions=evaluations * train_count, candidate_evaluations=candidate_evaluations,
               train_mse=-best.fitness, test_mse=test_mse, test_accuracy=test_accuracy,
               final_waltzer_train_mse="" if final is None else -final.fitness,
               final_waltzer_test_mse=final_mse, final_waltzer_test_accuracy=final_accuracy, training_seconds=seconds)
    diagnostic = dict(visits=visits, history=history, candidate_history=candidate_history,
                      generation_history=generation_history)
    network = neat.nn.FeedForwardNetwork.create(best, config)
    model = None
    if is_sine:
        grid = np.linspace(0, 1, 501).tolist()
        final_network = None if final is None else neat.nn.FeedForwardNetwork.create(final, config)
        diagnostic.update(grid=grid, best_curve=[network.activate((x,))[0] for x in grid],
                          final_curve=None if final_network is None else [final_network.activate((x,))[0] for x in grid])
    else:
        outputs = config.genome_config.output_keys
        model = dict(name=METHODS[method], network=network, train_mse=-best.fitness, test_mse=test_mse,
                     train_accuracy=classification_metrics(network, train)[1], test_accuracy=test_accuracy,
                     hidden_stored=sum(k not in outputs for k in best.nodes),
                     hidden_used=sum(k not in outputs for k, *_ in network.node_evals),
                     enabled=sum(c.enabled for c in best.connections.values()), links=len(best.connections))
    return row, diagnostic, model


def save_previews(prefix, task, trials, test, metadata):
    """Show every seed, without choosing examples by test performance."""
    seed = trials[0][0]["seed"]
    paths = []
    if task == "circle":
        path = Path(f"{prefix}_seed{seed}_boundaries.png")
        save_diagnostic(path, [model for _, _, model in trials], test, seed, metadata["generations"], 18,
                        initial_hidden=4, training_points=metadata["training_point_count"], mutation_power=0.1)
        paths.append(path)
    elif task == "spiral":
        from preview_spiral import save_previews as spiral_previews
        paths.extend(spiral_previews(Path(f"{prefix}_seed{seed}_boundaries"),
                                     [(row, model) for row, _, model in trials], metadata, comparison=True))
    else:
        figure = Figure(figsize=(12, 11), layout="constrained")
        for ax, (row, d, _) in zip(figure.subplots(3, 2).ravel(), trials):
            shade_regions(ax)
            ax.plot(d["grid"], [sine(x) for x in d["grid"]], "--", color="black", label="True function")
            ax.plot(d["grid"], d["best_curve"], color="#0072B2", label="Best training genome")
            if d["final_curve"] is not None:
                ax.plot(d["grid"], d["final_curve"], ":", color="#D55E00", label="Final waltzer")
            ax.set_title(f"{METHODS[row['method']]}\nTest MSE: {row['test_mse']:.5f}", fontsize=10)
            ax.legend(fontsize=7)
        figure.suptitle(f"Sine | evolution seed {seed} | {metadata['generations']} generations\n"
                        "Local selection for islands; common global training criterion for lineages and archive")
        path = Path(f"{prefix}_seed{seed}_predictions.png")
        figure.savefig(path, dpi=150)
        paths.append(path)

    figure = Figure(figsize=(11, 10 if task == "sine" else 8), layout="constrained")
    if task == "sine":
        errors, regions_panel, hosts, tempos = figure.subplots(4, 1, height_ratios=(2, 1.5, 1, 1))
    else:
        errors, hosts, tempos = figure.subplots(3, 1, height_ratios=(2, 1, 1))
    for row, d, _ in trials:
        events = [e for e in d["visits"] if e.get("waltzer", e.get("waltzer_id", 0)) == 0]
        if not events:
            continue
        xs = [events[0]["first_generation"] - 1] + [e["last_generation"] for e in events]
        ys = [-events[0]["before"]] + [-e["after"] for e in events]
        errors.step(xs, ys, where="post", label=METHODS[row["method"]])
        if row["method"] == "accelerating":
            hosts.step(xs, [e.get("host", e.get("island")) for e in [events[0], *events]], where="pre")
            tempos.step(xs, [e["tempo"] for e in [events[0], *events]], where="pre", color="#D55E00")
            if task == "sine":
                trace = [e for e in d["history"] if e["waltzer"] == 0]
                for r in range(4):
                    regions_panel.semilogy([e["generation"] for e in trace],
                                           [max(1e-12, e["region_mse"][r]) for e in trace], label=f"Region {chr(65 + r)}")
                regions_panel.set(ylabel="Progressive W1: regional MSE")
                regions_panel.legend(ncols=4, fontsize=8)
    for phase in metadata["tempo_phases"][:-1]:
        for ax in ((errors, regions_panel, hosts, tempos) if task == "sine" else (errors, hosts, tempos)):
            ax.axvline(phase["last_generation"] + 0.5, color="#888888", linestyle="--", linewidth=0.8)
    errors.set(ylabel="Training MSE of first waltzer", yscale="log")
    errors.legend(fontsize=8)
    hosts.set(ylabel="Host of progressive W1", yticks=range(4), yticklabels=list("ABCD"))
    tempos.set(ylabel="Generations per visit", yticks=(5, 10, 15), xlabel="Evaluated generation")
    figure.suptitle(f"{task.title()} | seed {seed} | traveling lineage and tempo\n"
                    "Dashed markers: phase transitions. Training traces only; test scores do not accept candidates.")
    path = Path(f"{prefix}_seed{seed}_progress.png")
    figure.savefig(path, dpi=150)
    paths.append(path)
    return paths


def save_summary(prefix, rows, task):
    """Prespecified paired contrasts, conditional on this task's fixed datasets."""
    summaries = []
    figure = Figure(figsize=(12, 10), layout="constrained")
    scores, effects = figure.subplots(2, 1)
    primary = "test_mse" if task == "sine" else "test_accuracy"
    for i, method in enumerate(METHODS):
        values = [r[primary] for r in sorted(rows, key=lambda r: r["seed"]) if r["method"] == method]
        scale = 1 if task == "sine" else 100
        values = np.array(values) * scale
        scores.scatter(values, i + np.linspace(-0.12, 0.12, len(values)), s=22)
        scores.scatter(np.median(values), i, marker="|", color="black", s=230)
    contrasts = [("best_genome", primary, m) for m in METHODS if m != "accelerating"]
    secondary = "final_waltzer_test_mse" if task == "sine" else "final_waltzer_test_accuracy"
    contrasts += [("final_waltzer", secondary, m) for m in ("fixed5", "fixed15", "stationary")]
    for i, (metric, field, baseline) in enumerate(contrasts):
        a = {r["seed"]: r[field] for r in rows if r["method"] == "accelerating"}
        b = {r["seed"]: r[field] for r in rows if r["method"] == baseline}
        gains = np.array([b[s] - a[s] if task == "sine" else (a[s] - b[s]) * 100 for s in sorted(a)])
        boot = np.median(np.random.default_rng(0).choice(gains, size=(10000, len(gains))), axis=1)
        low, high = np.quantile(boot, (0.025, 0.975))
        summary = dict(metric=metric, baseline=baseline, n=len(gains), median_gain=float(np.median(gains)),
                       ci_low=float(low), ci_high=float(high), wins=int(sum(gains > 1e-12)),
                       ties=int(sum(abs(gains) <= 1e-12)), losses=int(sum(gains < -1e-12)))
        summaries.append(summary)
        effects.scatter(gains, i + np.linspace(-0.12, 0.12, len(gains)), s=22)
        effects.hlines(i, low, high, color="black")
        effects.scatter(summary["median_gain"], i, marker="|", color="black", s=230)
        print(f"{task} | {metric} | progressive vs {baseline} | median gain {summary['median_gain']:+.6f} | "
              f"95% CI [{low:+.6f}, {high:+.6f}] | W/T/L {summary['wins']}/{summary['ties']}/{summary['losses']}")
    scores.set(yticks=range(len(METHODS)), yticklabels=list(METHODS.values()),
               xlabel="Test MSE (lower is better)" if task == "sine" else "Test accuracy (%)")
    effects.set(yticks=range(len(contrasts)),
                yticklabels=[f"{'Best genome' if metric == 'best_genome' else 'Final waltzer'} vs {METHODS[m]}"
                             for metric, _, m in contrasts],
                xlabel="MSE reduction" if task == "sine" else "Accuracy gain (percentage points)")
    effects.axvline(0, color="#888888", linestyle="--")
    for ax in (scores, effects):
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=8)
    figure.suptitle(f"{task.title()} | final tempo comparison\nPositive paired gains favor progressive Waltz")
    figure.supxlabel("Dots: evolution seeds; black marks: medians. Pointwise 95% bootstrap intervals, 10,000 resamples.\n"
                     "One fixed data pair per task; no multiplicity correction. Final waltzer is secondary.", fontsize=8)
    csv_path = Path(f"{prefix}.summary.csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    paths = [csv_path]
    for suffix in ("png", "svg"):
        path = Path(f"{prefix}_summary.{suffix}")
        figure.savefig(path, dpi=170)
        paths.append(path)
    return paths


def run_task(task, output, *, workers=3, smoke=False):
    generations, seeds = (121, (95,)) if smoke else (2000, tuple(range(95, 105)))
    schedule, phases = balanced_schedule(generations)
    train, test = make_data(task, smoke)
    train_flat = [x for region in train for x in region] if task == "sine" else train
    test_flat = [x for region in test for x in region] if task == "sine" else test
    if {x for x, _ in train_flat} & {x for x, _ in test_flat}:
        raise ValueError("Training/test inputs overlap")
    if any(Path(f"{output}{ext}").exists() for ext in (".csv", ".json")):
        raise FileExistsError(f"Existing results at {output}; select an unstarted --task or a new --output-dir")
    config = sine_config() if task == "sine" else create_config(4, mutation_power=0.1)
    settings = (NEAT_SETTINGS.replace("num_inputs = 2", f"num_inputs = {config.genome_config.num_inputs}")
                .replace("num_hidden = 0", "num_hidden = 4").replace("weight_mutate_power = 0.5", "weight_mutate_power = 0.1")
                .replace("bias_mutate_power = 0.5", "bias_mutate_power = 0.1"))
    metadata = dict(protocol="waltz_final_tempo_v1", task=task, stage="software_smoke" if smoke else "final_confirmation",
                    generations=generations, evolution_seeds=seeds, methods=METHODS, workers=workers,
                    population_total=120, island_count=4, island_population=30, waltzers=1 if task == "sine" else 4,
                    candidates_per_waltzer=1, initial_hidden=4, weight_mutate_power=0.1, bias_mutate_power=0.1,
                    initialization=INITIALIZATION, tempo_schedule=schedule, tempo_phases=phases,
                    schedule_rule="Nearest whole tours to cumulative one-third and two-thirds; final phase uses remainder",
                    historical_difference="Earlier 15->10->5 used one tour per tempo; this protocol allocates comparable phase durations",
                    stationary_control="Identical scheduled candidate/visit boundaries, no host changes",
                    primary_method="accelerating", primary_baselines=["fixed5", "fixed15", "stationary"],
                    secondary_baselines=["classic", "independent"],
                    primary_output="Best genome seen selected by global training MSE",
                    secondary_output="Best final waltzer selected by global training MSE; no test-based selection",
                    selection_scope="Local regional MSE for Sine island reproduction, global training MSE for lineage acceptance; centralized benchmark"
                        if task == "sine" else "Common global training MSE for population selection and lineage acceptance",
                    inference_scope="Conditional on one fixed dataset pair; pointwise bootstrap intervals, no multiplicity correction",
                    train_data_seed=22 if task == "sine" else 17, test_data_seed=23 if task == "sine" else 18,
                    training_point_count=len(train_flat), test_point_count=len(test_flat),
                    training_evaluations_per_run=generations * 120,
                    training_predictions_per_run=generations * 120 * len(train_flat),
                    training_examples=train, test_examples=test, neat_settings=settings,
                    bootstrap_seed=0, bootstrap_resamples=10000, python_version=platform.python_version(),
                    neat_python_version=version("neat-python"),
                    observation_schema="waltz_training_observations_v1",
                    observation_fitness="Higher is better; global for lineages/archive, local for Sine island selection",
                    source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                   for name in ("confirm_tempo.py", "waltz_tempo.py", "neat_waltz_domains.py", "neat_waltz.py",
                                                "neat_genome.py", "neat_migration.py", "test_sine.py", "test_xor.py",
                                                "test_circle.py", "test_spiral.py", "diagnose_circle.py", "preview_spiral.py",
                                                "compare_models.py", "plot_results.py")})
    output.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{output}.json").write_text(json.dumps(metadata, indent=2))
    jobs = [(task, train, test, generations, seed, method) for seed in seeds for method in METHODS]
    rows, pending, paths = [], {}, []
    print(f"Starting {task}: {len(jobs)} runs, {workers} processes, {generations * 120:,} evaluations per run.", flush=True)
    print(f"Tempo phases: {phases}", flush=True)
    with Path(f"{output}.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        stream.flush()
        pool = ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=get_context("spawn")) if workers > 1 else None
        try:
            results = (f.result() for f in as_completed([pool.submit(run_trial, *job) for job in jobs])) if pool else (run_trial(*job) for job in jobs)
            for trial in results:
                row, diagnostic, _ = trial
                writer.writerow(row)
                stream.flush()
                rows.append(row)
                Path(f"{output}_seed{row['seed']}_{row['method']}.json").write_text(json.dumps(dict(metrics=row, **diagnostic)))
                print(f"Completed {len(rows)}/{len(jobs)} | {task} | seed {row['seed']} | {METHODS[row['method']]} | "
                      f"test MSE {row['test_mse']:.6f} | {row['training_seconds']:.1f}s", flush=True)
                pending.setdefault(row["seed"], {})[row["method"]] = trial
                if len(pending[row["seed"]]) == len(METHODS):
                    new_paths = save_previews(output, task, [pending[row["seed"]][m] for m in METHODS], test, metadata)
                    paths.extend(new_paths)
                    del pending[row["seed"]]
                    for path in new_paths:
                        print(f"Saved {path}", flush=True)
        finally:
            if pool:
                pool.shutdown(cancel_futures=True)
    paths.extend(save_summary(output, rows, task))
    return rows, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--task", choices=("all", "circle", "spiral", "sine"), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("results/waltz_tempo_confirmation"))
    parser.add_argument("--smoke-test", action="store_true", help="Short software verification, not scientific evidence")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.smoke_test and args.output_dir == Path("results/waltz_tempo_confirmation"):
        args.output_dir = Path("results/waltz_tempo_smoke")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    tasks = ("circle", "spiral", "sine") if args.task == "all" else (args.task,)
    for task in tasks:
        if any((args.output_dir / f"{task}{ext}").exists() for ext in (".csv", ".json")):
            parser.error(f"Existing {task} results; choose an unstarted --task or a new --output-dir")
    for task in tasks:
        _, paths = run_task(task, args.output_dir / task, workers=args.workers, smoke=args.smoke_test)
        for path in paths:
            print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
