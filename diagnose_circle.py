"""Compare learned circle boundaries with python diagnose_circle.py."""

import argparse
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

import neat
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from neat_genome import HomologousGenome, INITIALIZATION
from neat_waltz import run_waltz
from test_circle import circle_examples
from test_xor import NEAT_SETTINGS, run_classic_neat


def create_config(hidden=0, *, mutation_power=0.5):
    """Use shared settings, initial hidden nodes and weight/bias mutation power."""
    if type(hidden) is not int or hidden < 0:
        raise ValueError("hidden must be a nonnegative integer")
    if not isfinite(mutation_power) or mutation_power < 0:
        raise ValueError("mutation_power must be finite and nonnegative")
    with TemporaryDirectory() as directory:
        path = Path(directory) / "neat.ini"
        path.write_text(NEAT_SETTINGS, encoding="utf-8")
        config = neat.Config(HomologousGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation, path)
    config.genome_config.num_hidden = hidden
    config.genome_config.weight_mutate_power = mutation_power
    config.genome_config.bias_mutate_power = mutation_power
    return config


def classification_metrics(network, examples):
    """Return MSE and accuracy with the existing threshold of 0.5."""
    predictions = [network.activate(inputs)[0] for inputs, _ in examples]
    mse = sum((prediction - target) ** 2
              for prediction, (_, target) in zip(predictions, examples)) / len(examples)
    accuracy = sum((prediction >= 0.5) == target
                   for prediction, (_, target) in zip(predictions, examples)) / len(examples)
    return mse, accuracy


def train_models(config, train, test, generations, seed, *, waltzers=4, tempo=5, candidates=1,
                 include_accelerating=False):
    """Select one genome per method on training fitness, then measure its behavior."""
    def fitness(genome, config):
        network = neat.nn.FeedForwardNetwork.create(genome, config)
        return -classification_metrics(network, train)[0]

    print(f"{generations * 120:,} training genome evaluations per method. "
          f"Classic NEAT: 120 genomes; independent/Waltz: 4 islands of 30. "
          f"Waltz: {waltzers} waltzer(s), {candidates} candidate(s) each, tempo {tempo}.", flush=True)
    print("Selection: best training fitness seen. Test and grid predictions are diagnostic only.", flush=True)
    print(f"Training points: {len(train)} | Test points: {len(test)}", flush=True)
    print(f"Initial hidden nodes per genome: {config.genome_config.num_hidden} | "
          f"Initial connections: {config.genome_config.initial_connection}", flush=True)
    print(f"Initialization: {INITIALIZATION}", flush=True)
    print("Gaussian mutation standard deviation (weights/biases): "
          f"{config.genome_config.weight_mutate_power:g}/{config.genome_config.bias_mutate_power:g}", flush=True)
    print("Method       | Train MSE | Test MSE | Train acc | Test acc | Hidden stored/used | Links enabled/total", flush=True)
    models = []
    names = ("Classic NEAT", "Independent", "Waltz")
    if include_accelerating:
        names += ("Waltz (15 → 10 → 5)",)
    for name in names:
        print(f"Training {name}...", flush=True)
        if name == "Classic NEAT":
            genome, model_config = run_classic_neat(config, fitness, generations, seed)
        else:
            options = {"tempo_schedule": (15, 10, 5)} if name == "Waltz (15 → 10 → 5)" else {"tempo": tempo}
            genome, _ = run_waltz(config, fitness, generations, seed=seed,
                                  waltzers=0 if name == "Independent" else waltzers,
                                  candidates_per_waltzer=candidates, **options)
            model_config = config
        network = neat.nn.FeedForwardNetwork.create(genome, model_config)
        train_mse, train_accuracy = classification_metrics(network, train)
        test_mse, test_accuracy = classification_metrics(network, test)
        outputs = model_config.genome_config.output_keys
        hidden_stored = sum(key not in outputs for key in genome.nodes)
        hidden_used = sum(key not in outputs for key, *_ in network.node_evals)
        enabled = sum(connection.enabled for connection in genome.connections.values())
        links = len(genome.connections)
        label = f"Waltz (fixed {tempo})" if include_accelerating and name == "Waltz" else name
        model = dict(name=label, network=network, train_mse=train_mse, test_mse=test_mse,
                     train_accuracy=train_accuracy, test_accuracy=test_accuracy,
                     hidden_stored=hidden_stored, hidden_used=hidden_used, enabled=enabled, links=links)
        models.append(model)
        print(f"{name:12} | {train_mse:9.6f} | {test_mse:8.6f} | {train_accuracy:9.2%} | "
              f"{test_accuracy:8.2%} | {hidden_stored:8}/{hidden_used:<8} | {enabled:7}/{links}", flush=True)
    return models


def save_diagnostic(path, models, test, seed, generations, test_seed, *, initial_hidden=0,
                    training_points=128, mutation_power=0.5):
    """Plot true labels and network outputs on identical grids and scales."""
    axis = np.linspace(-1, 1, 151)
    x, y = np.meshgrid(axis, axis)
    inputs = list(zip(x.ravel(), y.ravel()))
    truth = (x * x + y * y <= 0.8 ** 2).astype(float)
    points = np.array([point for point, _ in test])
    targets = np.array([target for _, target in test])
    cmap = LinearSegmentedColormap.from_list("circle_classes", ["#3291bf", "#f4f4f0", "#e68a35"])
    columns = 2
    rows = (len(models) + columns) // columns
    figure = Figure(figsize=(5.5 * columns, 5.5 * rows), layout="constrained")
    axes = np.atleast_1d(figure.subplots(rows, columns, sharex=True, sharey=True)).ravel()
    # Pair the two baselines and the two Waltz variants on their own rows.
    panels = [axes[0], *axes[2:]] if len(models) == 4 else list(axes[:len(models) + 1])
    figure.suptitle(f"Circle classification | evolution seed {seed}, test seed {test_seed}\n"
                    f"{generations} generations · {generations * 120:,} training evaluations per method\n"
                    f"Training points: {training_points} · test points: {len(test)}\n"
                    f"Initial hidden nodes per genome: {initial_hidden} · initial connections: full_direct\n"
                    f"Gaussian mutation standard deviation (weights/biases): {mutation_power:g}\n"
                    "Blue = outside (0) · orange = inside (1) · points = true test labels", fontsize=13)

    for panel, model in zip(panels, [None, *models]):
        if model is None:
            values = truth
            panel.set_title("True circle\nRadius 0.8 · reference labels", fontsize=11)
        else:
            values = np.array([model["network"].activate(point)[0] for point in inputs]).reshape(x.shape)
            if values.min() < 0.5 < values.max():
                panel.contour(x, y, values, levels=[0.5], colors="black", linewidths=1.4)
            else:
                panel.text(0.02, 0.02, "No 0.5 boundary in view", transform=panel.transAxes,
                           fontsize=9, bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"))
            predictions = np.array([model["network"].activate(point)[0] >= 0.5 for point in points])
            errors = points[predictions != targets]
            panel.scatter(errors[:, 0], errors[:, 1], marker="x", color="#bf1932", s=40,
                          linewidths=1.2, zorder=4)
            panel.set_title(f"{model['name']}\n"
                            f"Accuracy train/test: {model['train_accuracy']:.1%} / {model['test_accuracy']:.1%}\n"
                            f"MSE train/test: {model['train_mse']:.4f} / {model['test_mse']:.4f}\n"
                            f"Hidden stored/used: {model['hidden_stored']}/{model['hidden_used']} · "
                            f"Links enabled/total: {model['enabled']}/{model['links']}", fontsize=10)
        heatmap = panel.pcolormesh(x, y, values, cmap=cmap, vmin=0, vmax=1, shading="auto", zorder=0)
        panel.scatter(points[:, 0], points[:, 1], c=targets, cmap=cmap, vmin=0, vmax=1,
                      s=13, edgecolors="#333333", linewidths=0.3, zorder=3)
        panel.add_patch(Circle((0, 0), 0.8, fill=False, linestyle="--", color="black", linewidth=1.3))
        panel.set(xlim=(-1, 1), ylim=(-1, 1), aspect="equal", xlabel="x", ylabel="y")
    figure.colorbar(heatmap, ax=panels, shrink=0.7, label="Network output / reference label")
    legend = [Line2D([], [], color="black", linestyle="--", label="True boundary"),
              Line2D([], [], color="black", label="Learned boundary (output = 0.5)"),
              Line2D([], [], color="#bf1932", marker="x", linestyle="none", label="Test error")]
    if len(models) == 4:
        notes = axes[1]
        notes.set_axis_off()
        notes.text(0.1, 0.82, "Reading the comparison", transform=notes.transAxes, fontsize=15)
        notes.text(0.1, 0.65, "All methods use the same test points.\n"
                   "Point colors show the true classes.\n"
                   "Background colors show network outputs.\n\n"
                   "Genomes are selected by training MSE.\n"
                   "Test scores do not guide selection.",
                   transform=notes.transAxes, fontsize=11, va="top", linespacing=1.6)
        notes.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.08, 0.1), frameon=False, fontsize=10)
    else:
        for panel in axes[len(models) + 1:]:
            panel.set_visible(False)
        figure.legend(handles=legend, loc="outside lower center", ncols=3, fontsize=9)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=150)
    parser.add_argument("--seed", type=int, default=5, help="Evolution seed")
    parser.add_argument("--test-seed", type=int, default=11, help="Test data seed; training data stay at seed 10")
    parser.add_argument("--train-size", type=int, default=128,
                        help="Balanced training points, even and positive (default: 128)")
    parser.add_argument("--mutation-power", type=float, default=0.5,
                        help="Gaussian mutation standard deviation for weights and biases (default: 0.5)")
    parser.add_argument("--waltzers", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--tempo", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=0,
                        help="Initial hidden nodes in every genome for all three methods (default: 0)")
    parser.add_argument("--include-accelerating", action="store_true",
                        help="Also train and display Waltz with tempo 15->10->5")
    parser.add_argument("--output", type=Path, default=Path("circle_diagnostic.png"))
    args = parser.parse_args()
    if args.generations < 1 or args.tempo < 1 or not 1 <= args.waltzers <= 4 or not 1 <= args.candidates < 30:
        parser.error("Use positive generations/tempo, 1–4 waltzers and 1–29 candidates")
    if args.hidden < 0:
        parser.error("--hidden must be nonnegative")
    if args.train_size < 2 or args.train_size % 2:
        parser.error("--train-size must be a positive even number")
    if not isfinite(args.mutation_power) or args.mutation_power < 0:
        parser.error("--mutation-power must be finite and nonnegative")
    train = circle_examples(args.train_size // 2, seed=10)
    test = circle_examples(128, seed=args.test_seed)
    if {inputs for inputs, _ in train} & {inputs for inputs, _ in test}:
        parser.error("Training and test inputs overlap; choose a different --test-seed")
    models = train_models(create_config(args.hidden, mutation_power=args.mutation_power),
                          train, test, args.generations, args.seed,
                          waltzers=args.waltzers, tempo=args.tempo, candidates=args.candidates,
                          include_accelerating=args.include_accelerating)
    save_diagnostic(args.output, models, test, args.seed, args.generations, args.test_seed,
                    initial_hidden=args.hidden, training_points=len(train), mutation_power=args.mutation_power)
    print(f"Diagnostic graph saved to {args.output}")


if __name__ == "__main__":
    main()
