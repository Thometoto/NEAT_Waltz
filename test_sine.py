"""Explore a sine wave learned from four locally selected populations."""

import argparse
import csv
import hashlib
import json
import os
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from math import isfinite, pi, sin
from multiprocessing import get_context
from pathlib import Path
from random import Random
from time import perf_counter

import neat
import numpy as np
from matplotlib.figure import Figure

from diagnose_circle import create_config
from neat_genome import INITIALIZATION
from neat_waltz_domains import CROSSOVER_MODES, MODES, region_errors, run_domains
from test_xor import NEAT_SETTINGS


METHODS = dict(classic="Global NEAT", independent="Local islands", waltz="Travel, crossover + mutation",
               stationary="Stationary, crossover + mutation", mutation="Travel, mutation only", migration="Champion ring migration",
               waltz_no_mutation="Travel, crossover only", stationary_no_mutation="Stationary, crossover only")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
FIELDS = ("seed", "method", "generations", "genome_evaluations", "training_predictions",
          "candidate_evaluations", "migrant_evaluations", "train_mse", "test_mse",
          "final_waltzer_train_mse", "final_waltzer_test_mse", "training_seconds")


def sine(x):
    return (1 + sin(2 * pi * x)) / 2


def sine_regions(points_per_region, seed):
    """Independent uniform samples within each quarter; the target is never an input."""
    if type(points_per_region) is not int or points_per_region < 1:
        raise ValueError("points_per_region must be a positive integer")
    rng = Random(seed)
    regions = []
    for region in range(4):
        xs = [(region + rng.random()) / 4 for _ in range(points_per_region)]
        regions.append([((x,), sine(x)) for x in xs])
    return regions


def sine_config(hidden=4, mutation_power=0.1):
    config = create_config(hidden, mutation_power=mutation_power)
    config.genome_config.num_inputs = 1
    config.genome_config.input_keys = [-1]
    return config


def train_trial(config, train, test, generations, seed, method, tempo, waltzers):
    """Training receives no test data; diagnostics run only after evolution ends."""
    print(f"Seed {seed} | {METHODS[method]} | {generations * 120:,} training genome evaluations...", flush=True)
    start = perf_counter()
    result = run_domains(config, train, generations, mode=method, seed=seed, tempo=tempo, waltzers=waltzers)
    seconds = perf_counter() - start
    grid = np.linspace(0, 1, 501)

    def network(genome):
        return neat.nn.FeedForwardNetwork.create(genome, config)

    def curve(genome):
        model = network(genome)
        return [model.activate((float(x),))[0] for x in grid]

    best = result["best"]
    network_cache = network(best)
    test_errors = region_errors(network_cache, test)
    final = max(result["lineages"], key=lambda g: g.fitness, default=None)
    row = dict(seed=seed, method=method, generations=generations,
               **{key: result[key] for key in ("genome_evaluations", "training_predictions",
                                             "candidate_evaluations", "migrant_evaluations")},
               train_mse=-best.fitness, test_mse=sum(test_errors) / 4,
               final_waltzer_train_mse="" if final is None else -final.fitness,
               final_waltzer_test_mse="" if final is None else sum(region_errors(network(final), test)) / 4,
               training_seconds=seconds)
    diagnostic = dict(seed=seed, method=method, grid=grid.tolist(), best_curve=curve(best),
                      train_mse_by_region=list(best.region_mse), test_mse_by_region=list(test_errors),
                      final_waltzer_curve=None, snapshots=[], history=result["history"], visits=result["visits"])
    if final is not None:
        diagnostic["final_waltzer_curve"] = curve(final)
    for generation, genome in result["snapshots"]:
        diagnostic["snapshots"].append(dict(generation=generation, curve=curve(genome)))
    return row, diagnostic


def shade_regions(axes):
    for region, color in enumerate(COLORS):
        axes.axvspan(region / 4, (region + 1) / 4, color=color, alpha=0.07)
        axes.axvline(region / 4, color="#aaaaaa", linewidth=0.6)
    axes.set(xlim=(0, 1), ylim=(-0.05, 1.05), xlabel="x", ylabel="y")


def save_previews(prefix, trials):
    """Plot all methods, plus the prespecified first waltzer's learning trajectory."""
    seed = trials[0][0]["seed"]
    panel_rows = (len(trials) + 1) // 2
    figure = Figure(figsize=(12, 3.3 * panel_rows + 0.7), layout="constrained")
    axes = figure.subplots(panel_rows, 2).ravel()
    for ax, (row, diagnostic) in zip(axes, trials):
        grid = diagnostic["grid"]
        shade_regions(ax)
        ax.plot(grid, [sine(x) for x in grid], "--", color="black", label="True function")
        ax.plot(grid, diagnostic["best_curve"], color="#0072B2", label="Best by global training MSE")
        if diagnostic["final_waltzer_curve"] is not None:
            ax.plot(grid, diagnostic["final_waltzer_curve"], ":", color="#9467BD", label="Best final waltzer")
        ax.set_title(f"{METHODS[row['method']]}\nTrain/test MSE: {row['train_mse']:.5f} / {row['test_mse']:.5f}", fontsize=10)
        ax.legend(fontsize=7, loc="lower left")
    figure.suptitle(f"Sine regression | evolution seed {seed} | {trials[0][0]['generations']} generations\n"
                    "Local populations: A [0, .25), B [.25, .5), C [.5, .75), D [.75, 1]\n"
                    "Global training criterion selects outputs and lineage replacements; test is diagnostic only.", fontsize=12)
    paths = []
    path = Path(f"{prefix}_seed{seed}_predictions.png")
    figure.savefig(path, dpi=150)
    paths.append(path)
    for method in ("waltz", "waltz_no_mutation"):
        trial = next((trial for trial in trials if trial[0]["method"] == method), None)
        if trial is not None:
            paths.append(save_lineage_preview(prefix, *trial))
    return paths


def save_lineage_preview(prefix, row, traveling):
    """Use the same training-only progress view for each mobile crossover rule."""
    seed, method = row["seed"], row["method"]
    figure = Figure(figsize=(11, 10), layout="constrained")
    curves, errors, hosts = figure.subplots(3, 1, height_ratios=(1.2, 1, 0.35))
    shade_regions(curves)
    curves.plot(traveling["grid"], [sine(x) for x in traveling["grid"]], "--", color="black", label="True function")
    for snapshot in traveling["snapshots"]:
        curves.plot(traveling["grid"], snapshot["curve"], label=f"Generation {snapshot['generation']}")
    curves.legend(ncols=3, fontsize=8)
    curves.set_title("First waltzer: initialization, complete tours and final generation", fontsize=11)
    trace = [event for event in traveling["history"] if event["waltzer"] == 0]
    rounds = [event["generation"] for event in trace]
    for region, color in enumerate(COLORS):
        errors.semilogy(rounds, [max(1e-12, event["region_mse"][region]) for event in trace],
                        color=color, label=f"Region {chr(65 + region)}")
    errors.semilogy(rounds, [max(1e-12, event["global_train_mse"]) for event in trace],
                    color="black", linewidth=2, label="Global training MSE")
    errors.set(xlabel="Evaluated generation (0 = initialization)", ylabel="Training MSE (log scale)")
    errors.legend(ncols=3, fontsize=8)
    hosts.step(rounds, [event["host"] for event in trace], where="post", color="#333333")
    hosts.set(yticks=range(4), yticklabels=list("ABCD"), ylim=(-0.3, 3.3),
              xlabel="Evaluated generation", ylabel="Host")
    figure.suptitle(f"{METHODS[method]} | W1 | evolution seed {seed}\n"
                    "Acceptance protects mean training MSE; individual regions can still regress.", fontsize=12)
    suffix = "waltzer" if method == "waltz" else "waltzer_no_mutation"
    path = Path(f"{prefix}_seed{seed}_{suffix}.png")
    figure.savefig(path, dpi=150)
    return path


def save_summary(prefix, rows):
    """Positive paired gains mean lower test MSE for the explicitly compared method."""
    modes = CROSSOVER_MODES if any(r["method"] == "waltz_no_mutation" for r in rows) else MODES
    grouped = {mode: {int(r["seed"]): float(r["test_mse"]) for r in rows if r["method"] == mode} for mode in modes}
    final = {mode: {int(r["seed"]): float(r["final_waltzer_test_mse"]) for r in rows
                    if r["method"] == mode and r["final_waltzer_test_mse"] != ""}
             for mode in ("waltz", "stationary", "mutation", "waltz_no_mutation", "stationary_no_mutation")}
    summaries = []
    if "waltz_no_mutation" in modes:
        contrasts = [(metric, target, baseline) for metric in ("final_lineage", "best_genome")
                     for target, baseline in (("waltz_no_mutation", "waltz"),
                                              ("stationary_no_mutation", "stationary"),
                                              ("waltz_no_mutation", "stationary_no_mutation"))]
        contrasts += [("final_lineage", "waltz_no_mutation", "mutation"),
                      ("best_genome", "waltz_no_mutation", "classic")]
    else:
        contrasts = [("best_genome", "waltz", baseline) for baseline in ("stationary", "mutation", "migration", "classic", "independent")]
        contrasts += [("final_lineage", "waltz", baseline) for baseline in ("stationary", "mutation")]
    for metric, method, baseline in contrasts:
        groups = grouped if metric == "best_genome" else final
        seeds = sorted(groups[method].keys() & groups[baseline].keys())
        gains = np.array([groups[baseline][s] - groups[method][s] for s in seeds])
        low = high = ""
        if len(seeds) > 1:
            bootstrap = np.median(np.random.default_rng(0).choice(gains, size=(10000, len(seeds)), replace=True), axis=1)
            low, high = map(float, np.quantile(bootstrap, (0.025, 0.975)))
        summaries.append(dict(metric=metric, method=method, baseline=baseline, n=len(seeds), median_mse_gain=float(np.median(gains)),
                              ci_low=low, ci_high=high, wins=int(sum(gains > 1e-12)),
                              ties=int(sum(abs(gains) <= 1e-12)), losses=int(sum(gains < -1e-12))))
    path = Path(f"{prefix}.summary.csv")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    paths = [path]
    figure = Figure(figsize=(11, 11), layout="constrained")
    scores, effects = figure.subplots(2, 1)
    for i, mode in enumerate(modes):
        values = [grouped[mode][seed] for seed in sorted(grouped[mode])]
        scores.scatter(values, i + np.linspace(-0.1, 0.1, len(values)), color="#0072B2", s=25)
        scores.scatter(np.median(values), i, marker="|", color="black", s=250)
    scores.set(yticks=range(len(modes)), yticklabels=[METHODS[m] for m in modes], xlabel="Test MSE (lower is better)")
    scores.invert_yaxis()
    for i, summary in enumerate(summaries):
        baseline = summary["baseline"]
        groups = grouped if summary["metric"] == "best_genome" else final
        method = summary["method"]
        gains = [groups[baseline][s] - groups[method][s] for s in sorted(groups[method])]
        effects.scatter(gains, i + np.linspace(-0.1, 0.1, len(gains)), s=25, color="#009E73")
        effects.scatter(summary["median_mse_gain"], i, marker="|", color="black", s=250)
        if summary["ci_low"] != "":
            effects.hlines(i, summary["ci_low"], summary["ci_high"], color="black")
        interval = "not estimated" if summary["ci_low"] == "" else f"[{summary['ci_low']:+.6f}, {summary['ci_high']:+.6f}]"
        print(f"{summary['metric']} | {METHODS[method]} vs {METHODS[baseline]} | n={summary['n']} | median MSE reduction "
              f"{summary['median_mse_gain']:+.6f} | 95% CI {interval} | "
              f"W/T/L {summary['wins']}/{summary['ties']}/{summary['losses']}")
    effects.axvline(0, color="#888888", linestyle="--")
    effects.set(yticks=range(len(summaries)),
                yticklabels=[f"{'Genome' if s['metric'] == 'best_genome' else 'Lineage'}: {METHODS[s['method']]}\nvs {METHODS[s['baseline']]}" for s in summaries],
                xlabel="Baseline MSE minus compared-method MSE (positive favors the compared method)")
    effects.tick_params(axis="y", labelsize=8)
    effects.invert_yaxis()
    figure.suptitle("Sine regression | exploratory domain-specialization experiment", fontsize=13)
    figure.supxlabel("Dots: evolution seeds; black ticks: medians. Pointwise 95% bootstrap intervals, 10,000 resamples.\n"
                     "One fixed training/test pair. These runs are exploratory; test data never guide training.", fontsize=8)
    for extension in ("png", "svg"):
        path = Path(f"{prefix}_summary.{extension}")
        figure.savefig(path, dpi=180)
        paths.append(path)
    return paths


def run_experiment(output, *, generations=500, seeds=range(4), workers=3, tempo=5, waltzers=1,
                   train_per_region=128, test_per_region=256, train_seed=20, test_seed=21,
                   hidden=4, mutation_power=0.1, compare_crossover=False):
    """Save protocol before training; write each finished trial immediately."""
    seeds = tuple(seeds)
    if (not seeds or len(set(seeds)) != len(seeds) or workers < 1 or generations < 1 or tempo < 1
            or not 1 <= waltzers <= 4 or hidden < 0 or not isfinite(mutation_power) or mutation_power < 0):
        raise ValueError("Use distinct seeds, positive workers/generations/tempo, 1–4 waltzers and valid NEAT settings")
    output = Path(output)
    if any(Path(f"{output}{suffix}").exists() for suffix in (".csv", ".json")):
        raise FileExistsError("Output exists; use a new --output prefix")
    train, test = sine_regions(train_per_region, train_seed), sine_regions(test_per_region, test_seed)
    if {x for r in train for x, _ in r} & {x for r in test for x, _ in r}:
        raise ValueError("Training and test inputs overlap; choose another test seed")
    config = sine_config(hidden, mutation_power)
    modes = CROSSOVER_MODES if compare_crossover else MODES
    settings = (NEAT_SETTINGS.replace("num_inputs = 2", "num_inputs = 1")
                .replace("num_hidden = 0", f"num_hidden = {hidden}")
                .replace("weight_mutate_power = 0.5", f"weight_mutate_power = {mutation_power}")
                .replace("bias_mutate_power = 0.5", f"bias_mutate_power = {mutation_power}"))
    metadata = dict(protocol="sine_crossover_v1" if compare_crossover else "sine_domains_v1",
                    stage="exploratory", target="(1 + sin(2*pi*x))/2", domain=[0, 1],
                    generations=generations, evolution_seeds=seeds, workers=workers, tempo=tempo, waltzers=waltzers,
                    starting_hosts="(evolution_seed + lineage_index) mod 4", population_total=120, island_population=30,
                    train_data_seed=train_seed, test_data_seed=test_seed, training_regions=train, test_regions=test,
                    initialization=INITIALIZATION, neat_settings=settings, methods={m: METHODS[m] for m in modes},
                    training_evaluations_per_run=generations * 120,
                    training_predictions_per_run=generations * 120 * 4 * train_per_region,
                    primary_metric="Test MSE of the best genome selected by global training MSE",
                    secondary_metric="Test MSE of the best final lineage selected by global training MSE",
                    primary_controls=["stationary", "mutation", "migration"] if not compare_crossover else None,
                    primary_contrasts=None if not compare_crossover else [
                        dict(metric="final_lineage", method=a, baseline=b) for a, b in (
                            ("waltz_no_mutation", "waltz"), ("stationary_no_mutation", "stationary"),
                            ("waltz_no_mutation", "stationary_no_mutation"))],
                    candidate_mutation={m: m not in ("waltz_no_mutation", "stationary_no_mutation")
                                        for m in modes if m not in ("classic", "independent", "migration")},
                    ordinary_population_mutation="Unchanged in every mode",
                    local_selection="Own-region MSE only; global NEAT uses mean MSE over four regions",
                    global_selection="All-region training MSE for output archive, lineage acceptance and both crossover parents",
                    budget="Every evaluated genome predicts all training points once; all global/local scores reuse these predictions",
                    interpretation="Centralized global training gate; not a data-private or local-only learning experiment",
                    forgetting="Mean training error cannot increase along a lineage, but individual region errors can",
                    bootstrap_resamples=10000, bootstrap_seed=0,
                    python_version=platform.python_version(), neat_python_version=version("neat-python"),
                    source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                   for name in ("test_sine.py", "neat_waltz_domains.py", "neat_waltz.py", "neat_migration.py",
                                                "neat_genome.py", "diagnose_circle.py", "test_xor.py")})
    output.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{output}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    jobs = [(config, train, test, generations, seed, mode, tempo, waltzers) for seed in seeds for mode in modes]
    rows, pending, paths = [], {}, []
    workers = min(workers, len(jobs))
    print(f"Sine regression | {len(jobs)} trials | {workers} worker(s) | {generations * 120:,} genome evaluations "
          f"and {metadata['training_predictions_per_run']:,} training predictions per run.", flush=True)
    with Path(f"{output}.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        stream.flush()
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) if workers > 1 else None
        try:
            results = (future.result() for future in as_completed([pool.submit(train_trial, *job) for job in jobs])) if pool else (train_trial(*job) for job in jobs)
            for row, diagnostic in results:
                rows.append(row)
                writer.writerow(row)
                stream.flush()
                path = Path(f"{output}_seed{row['seed']}_{row['method']}.json")
                path.write_text(json.dumps(dict(metrics=row, **diagnostic), indent=2), encoding="utf-8")
                print(f"Completed {len(rows)}/{len(jobs)} | seed {row['seed']} | {METHODS[row['method']]} | "
                      f"test MSE {row['test_mse']:.6f} | {row['training_seconds']:.1f}s", flush=True)
                if row["final_waltzer_test_mse"] != "":
                    print(f"  Final lineage | train/test MSE {row['final_waltzer_train_mse']:.6f} / "
                          f"{row['final_waltzer_test_mse']:.6f}", flush=True)
                pending.setdefault(row["seed"], {})[row["method"]] = (row, diagnostic)
                if len(pending[row["seed"]]) == len(modes):
                    paths.extend(save_previews(output, [pending[row["seed"]][mode] for mode in modes]))
                    del pending[row["seed"]]
                    print(f"Prediction and waltzer previews saved for seed {row['seed']}", flush=True)
        finally:
            if pool:
                pool.shutdown(cancel_futures=True)
    paths.extend(save_summary(output, rows))
    return rows, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=500)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--seeds", type=int, nargs=2, default=(0, 3), metavar=("FIRST", "LAST"))
    choice.add_argument("--seed", type=int, help="One evolution seed instead of a range")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--tempo", type=int, default=5)
    parser.add_argument("--waltzers", type=int, default=1)
    parser.add_argument("--train-per-region", type=int, default=128)
    parser.add_argument("--test-per-region", type=int, default=256)
    parser.add_argument("--train-seed", type=int, default=20)
    parser.add_argument("--test-seed", type=int, default=21)
    parser.add_argument("--hidden", type=int, default=4)
    parser.add_argument("--mutation-power", type=float, default=0.1)
    parser.add_argument("--compare-crossover", action="store_true", help="Add mobile/fixed crossover-only candidates; keep ordinary NEAT mutation")
    parser.add_argument("--output", type=Path, default=Path("results/sine_domains/pilot"))
    args = parser.parse_args()
    if args.compare_crossover and args.output == Path("results/sine_domains/pilot"):
        args.output = Path("results/sine_domains/crossover_ablation")
    if args.seeds[0] > args.seeds[1]:
        parser.error("FIRST must be <= LAST")
    seeds = (args.seed,) if args.seed is not None else range(args.seeds[0], args.seeds[1] + 1)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    try:
        _, paths = run_experiment(args.output, generations=args.generations, seeds=seeds, workers=args.workers,
                                 tempo=args.tempo, waltzers=args.waltzers, train_per_region=args.train_per_region,
                                 test_per_region=args.test_per_region, train_seed=args.train_seed, test_seed=args.test_seed,
                                 hidden=args.hidden, mutation_power=args.mutation_power, compare_crossover=args.compare_crossover)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    for path in (Path(f"{args.output}.csv"), Path(f"{args.output}.json"), *paths):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
