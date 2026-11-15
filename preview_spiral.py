"""Replay selected best Spiral trials and plot their learned decision regions."""

import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from math import isclose
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory

import neat
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from compare_models import METHOD_IDS, _run_trial
from neat_genome import HomologousGenome


def select_best(rows, method, count):
    """Rank by test accuracy, then training MSE, then seed; selection is post hoc."""
    rows = [row for row in rows if row["method"] == method]
    return sorted(rows, key=lambda row: (-float(row["test_accuracy"]),
                                         float(row["train_mse"]), int(row["seed"])))[:count]


def select_seed(rows, seed):
    """Compare the four methods at one seed in the original panel order."""
    selected = [row for row in rows if int(row["seed"]) == seed and row["method"] in METHOD_IDS]
    if len(selected) != len(METHOD_IDS) or {row["method"] for row in selected} != set(METHOD_IDS):
        raise ValueError(f"Seed {seed} needs exactly one completed trial for each of the four methods")
    return [next(row for row in selected if row["method"] == method) for method in METHOD_IDS]


def replay_trial(row, metadata):
    """Recreate the saved protocol and require the original scores to match."""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "neat.ini"
        path.write_text(metadata["neat_settings"], encoding="utf-8")
        config = neat.Config(HomologousGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation, path)
    actual, model = _run_trial(config, metadata["training_examples"], metadata["test_examples"],
                               int(row["generations"]), int(row["seed"]), row["method"],
                               metadata["waltzers"], metadata["candidates_per_waltzer"], True)
    for field in ("train_mse", "test_mse", "test_accuracy", "genome_evaluations",
                  "hidden_used", "enabled_connections"):
        if not isclose(float(actual[field]), float(row[field]), rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(f"Replay mismatch for seed {row['seed']}, {field}")
    return actual, model


def save_previews(output, trials, metadata, *, comparison=False):
    """Show true data support, not invented ground-truth labels between arms."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    axis = np.linspace(-1.05, 1.05, 201)
    x, y = np.meshgrid(axis, axis)
    grid = list(zip(x.ravel(), y.ravel()))
    points = np.array([point for point, _ in metadata["test_examples"]])
    labels = np.array([label for _, label in metadata["test_examples"]])
    position = np.linspace(0, 1, 1000)
    radius = 0.15 + 0.85 * position
    cmap = LinearSegmentedColormap.from_list("spiral_classes", ["#3291bf", "#f4f4f0", "#e68a35"])
    fields = [np.array([model["network"].activate(point)[0] for point in grid]).reshape(x.shape)
              for _, model in trials]

    def draw(panel, index=None):
        if index is None:
            panel.set_title("Reference data\nTwo noiseless spiral arms", fontsize=12)
        else:
            row, model = trials[index]
            values = fields[index]
            heatmap = panel.pcolormesh(x, y, values, cmap=cmap, vmin=0, vmax=1, shading="auto")
            if values.min() < 0.5 < values.max():
                panel.contour(x, y, values, levels=[0.5], colors="black", linewidths=1.3)
            predictions = np.array([model["network"].activate(point)[0] >= 0.5 for point in points])
            errors = points[predictions != labels]
            panel.scatter(errors[:, 0], errors[:, 1], marker="x", color="#b71931", s=45, zorder=5)
            title = model["name"] if comparison else f"{model['name']} | seed {row['seed']}"
            panel.set_title(f"{title}\n"
                            f"Accuracy train/test: {model['train_accuracy']:.1%} / {model['test_accuracy']:.1%}\n"
                            f"MSE train/test: {model['train_mse']:.5f} / {model['test_mse']:.5f}\n"
                            f"Hidden stored/used: {model['hidden_stored']}/{model['hidden_used']} | "
                            f"Links enabled/total: {model['enabled']}/{model['links']}", fontsize=10)
        for label in (0, 1):
            angle = 2 * np.pi * position + label * np.pi
            panel.plot(radius * np.cos(angle), radius * np.sin(angle), "--", color="#333333",
                       linewidth=0.9, alpha=0.8, zorder=2)
        panel.scatter(points[:, 0], points[:, 1], c=labels, cmap=cmap, vmin=0, vmax=1,
                      s=14, edgecolors="#333333", linewidths=0.3, zorder=3)
        panel.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), aspect="equal", xlabel="x", ylabel="y")
        return heatmap if index is not None else None

    legend = [Line2D([], [], color="#333333", linestyle="--", label="Data curves (not class boundaries)"),
              Line2D([], [], color="black", label="Learned boundary (output = 0.5)"),
              Line2D([], [], marker="x", color="#b71931", linestyle="none", label="Test error")]
    rows = (len(trials) + 3) // 2 if comparison else (len(trials) + 2) // 2
    figure = Figure(figsize=(12, 5.5 * rows), layout="constrained")
    axes = np.atleast_1d(figure.subplots(rows, 2)).ravel()
    draw(axes[0])
    panels = list(axes[2:2 + len(trials)]) if comparison else list(axes[1:len(trials) + 1])
    for index in range(len(trials)):
        heatmap = draw(panels[index], index)
    if comparison:
        for panel in axes[2 + len(trials):]:
            panel.set_visible(False)
        notes = axes[1]
        notes.set_axis_off()
        notes.text(0.08, 0.90, "Reading the comparison", transform=notes.transAxes, fontsize=15)
        notes.text(0.08, 0.77, "All methods use the same test points.\n"
                   "Point colors show the true classes.\n"
                   "Background colors show network outputs.\n\n"
                   "Genomes are selected by training MSE.\n"
                   "This seed is shown for illustration.\n\n"
                   "Labels between the arms are undefined.",
                   transform=notes.transAxes, fontsize=11, va="top", linespacing=1.6)
        notes.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.04, 0.02), frameon=False, fontsize=9)
        figure.suptitle(f"Spiral classification | evolution seed {trials[0][0]['seed']}, "
                        f"test seed {metadata['test_data_seed']}\n"
                        f"{metadata['generations']:,} generations | "
                        f"{metadata['training_evaluations_per_run']:,} training evaluations per method\n"
                        f"Training points: {len(metadata['training_examples'])} | test points: {len(points)}\n"
                        f"Initial hidden nodes per genome: {metadata['initial_hidden']} | initial connections: full_direct\n"
                        f"Gaussian mutation standard deviation (weights/biases): {metadata['weight_mutate_power']:g}\n"
                        "Blue = class 0 | orange = class 1 | points = true test labels", fontsize=13)
    else:
        for panel in axes[len(trials) + 1:]:
            panel.set_visible(False)
        figure.suptitle(f"Spiral | selected best trials | {metadata['generations']:,} generations\n"
                        f"Training/test points: {len(metadata['training_examples'])}/{len(points)} | "
                        f"test seed {metadata['test_data_seed']}\n"
                        "Blue = class 0 | orange = class 1 | background = network output\n"
                        "Best cases selected by test accuracy; labels between the arms are undefined.", fontsize=12)
        figure.legend(handles=legend, loc="outside lower center", ncols=1, fontsize=10)
    figure.colorbar(heatmap, ax=panels, shrink=0.65, label="Network output (class 1)")
    combined = Path(f"{output}.png")
    figure.savefig(combined, dpi=170)
    paths = [combined]
    if comparison:
        return paths
    for index, (row, _) in enumerate(trials):
        single = Figure(figsize=(7, 7), layout="constrained")
        panel = single.subplots()
        heatmap = draw(panel, index)
        single.colorbar(heatmap, ax=panel, shrink=0.75, label="Network output (class 1)")
        single.suptitle("Spiral | selected best case\nLabels between the arms are undefined.", fontsize=12)
        single.legend(handles=legend, loc="outside lower center", fontsize=9)
        path = Path(f"{output}_seed{row['seed']}.png")
        single.savefig(path, dpi=170)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/waltz_final_confirmation/spiral.csv"))
    parser.add_argument("--method", choices=("waltz_fixed_5", "waltz_accelerating"), default="waltz_fixed_5")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, help="Compare all four methods at this evolution seed, using the Circle layout")
    parser.add_argument("--output", type=Path, help="Output prefix; default is chosen from the mode and results path")
    args = parser.parse_args()
    if args.count < 1 or args.workers < 1:
        parser.error("--count and --workers must be positive")
    metadata_path = args.results.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("task", "").lower() != "spiral":
        parser.error("--results must refer to a Spiral experiment")
    root = Path(__file__).resolve().parent
    for name, digest in metadata["source_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            parser.error(f"Source changed since training: {name}")
    if version("neat-python") != metadata["neat_python_version"]:
        parser.error("neat-python version differs from the original experiment")
    with args.results.open() as stream:
        rows = list(csv.DictReader(stream))
    if args.seed is None:
        selected = select_best(rows, args.method, args.count)
        default_name = "spiral_best_fixed_previews" if args.method == "waltz_fixed_5" else "spiral_best_accelerating_previews"
        selection_rule = "Descending test accuracy, ascending training MSE, ascending seed; post hoc best cases"
    else:
        try:
            selected = select_seed(rows, args.seed)
        except ValueError as error:
            parser.error(str(error))
        default_name = f"spiral_seed{args.seed}_boundaries"
        selection_rule = f"All four methods at explicitly selected evolution seed {args.seed}; illustrative comparison"
    output = args.output or args.results.parent / default_name
    if not selected:
        parser.error("No completed trials for this method")
    if Path(f"{output}.json").exists() or Path(f"{output}.png").exists():
        parser.error("Preview output exists; choose a new --output prefix")
    print("Replaying selected trials:", [(int(row["seed"]), row["method"]) for row in selected], flush=True)
    trials = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(selected)), mp_context=get_context("spawn")) as pool:
        jobs = {pool.submit(replay_trial, row, metadata): row for row in selected}
        for job in as_completed(jobs):
            row, model = job.result()
            trials[int(row["seed"]), row["method"]] = row, model
            print(f"Replay verified: seed {row['seed']} | {model['name']} | test {row['test_accuracy']:.2%}", flush=True)
    ordered = [trials[int(row["seed"]), row["method"]] for row in selected]
    paths = save_previews(output, ordered, metadata, comparison=args.seed is not None)
    provenance = dict(source_results=str(args.results.resolve()), source_metadata=str(metadata_path.resolve()),
                      selection=selection_rule,
                      original_scores_verified=True, original_rows=selected,
                      source_sha256=metadata["source_sha256"],
                      preview_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    Path(f"{output}.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    for path in paths:
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
