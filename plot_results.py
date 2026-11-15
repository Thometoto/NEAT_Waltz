"""Plot saved comparison results: python plot_results.py results/example.csv."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure


METHODS = {
    "classic_neat": ("Classic NEAT", "#666666"),
    "independent": ("Independent islands", "#0072B2"),
    "waltz_fixed_5": ("Waltz, fixed 5", "#009E73"),
    "waltz_fixed_15": ("Waltz, fixed 15", "#9467BD"),
    "waltz_accelerating": ("Waltz, 15 → 10 → 5", "#D55E00"),
    "waltz_stationary": ("Waltz, stationary", "#9467BD"),
    "ring_migration": ("Champion ring migration", "#CC79A7"),
}
PAIRS = (
    ("waltz_fixed_5", "classic_neat", "Fixed 5 − NEAT"),
    ("waltz_fixed_5", "independent", "Fixed 5 − islands"),
    ("waltz_accelerating", "classic_neat", "Accelerating − NEAT"),
    ("waltz_accelerating", "independent", "Accelerating − islands"),
    ("waltz_accelerating", "waltz_fixed_5", "Accelerating − fixed 5"),
    ("waltz_accelerating", "waltz_fixed_15", "Accelerating − fixed 15"),
    ("waltz_fixed_5", "waltz_stationary", "Travel − stationary"),
    ("waltz_fixed_5", "ring_migration", "Travel − ring migration"),
)


def read_results(path):
    """Require one finite accuracy per method, generation budget and seed."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    for row in rows:
        row["seed"], row["generations"] = int(row["seed"]), int(row["generations"])
        row["test_accuracy"] = float(row["test_accuracy"])
        key = row["generations"], row["method"], row["seed"]
        if key in seen or row["method"] not in METHODS:
            raise ValueError(f"Duplicate or unknown method: {key}")
        if row["generations"] < 1 or not 0 <= row["test_accuracy"] <= 1:
            raise ValueError(f"Invalid generation budget or accuracy: {key}")
        seen.add(key)
    if not rows:
        raise ValueError("No completed runs in this CSV")
    return rows


def paired_summary(rows):
    """Bootstrap the median of paired seed differences on a fixed dataset."""
    groups = defaultdict(dict)
    for row in rows:
        groups[row["generations"], row["method"]][row["seed"]] = row["test_accuracy"]
    summaries = []
    for generations in sorted({row["generations"] for row in rows}):
        for method, baseline, label in PAIRS:
            a, b = groups[generations, method], groups[generations, baseline]
            seeds = sorted(a.keys() & b.keys())
            if not seeds:
                continue
            gains = np.array([100 * (a[seed] - b[seed]) for seed in seeds])
            low = high = None
            if len(seeds) > 1:
                samples = np.random.default_rng(0).choice(gains, size=(10000, len(seeds)), replace=True)
                low, high = map(float, np.quantile(np.median(samples, axis=1), [0.025, 0.975]))
            summaries.append(dict(generations=generations, method=method, baseline=baseline,
                                  label=label, seeds=seeds, gains=gains.tolist(), n=len(seeds),
                                  median_gain_pp=float(np.median(gains)), ci_low_pp=low, ci_high_pp=high,
                                  wins=int(sum(gains > 1e-10)), ties=int(sum(abs(gains) <= 1e-10)),
                                  losses=int(sum(gains < -1e-10))))
    return summaries


def save_plots(csv_path):
    """Save score distributions, paired effects, vector figures and summaries."""
    csv_path = Path(csv_path)
    prefix = csv_path.with_suffix("")
    rows = read_results(csv_path)
    metadata_path = Path(f"{prefix}.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    summaries = paired_summary(rows)
    summary_path = Path(f"{prefix}.summary.csv")
    fields = ("generations", "method", "baseline", "n", "median_gain_pp", "ci_low_pp", "ci_high_pp", "wins", "ties", "losses")
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)

    paths = [summary_path]
    for generations in sorted({row["generations"] for row in rows}):
        current = [row for row in rows if row["generations"] == generations]
        effects = [item for item in summaries if item["generations"] == generations]
        figure = Figure(figsize=(9, 10), layout="constrained")
        scores, differences = figure.subplots(2, 1, height_ratios=(1, 1.2))
        task, stage = metadata.get("task", "Classification"), metadata.get("stage", "Saved runs")
        title = (f"{task} — {stage}\n{generations:,} generations, "
                 f"{generations * metadata.get('population_total', 120):,} training evaluations per run")
        if "initial_hidden" in metadata:
            title += f"\nInitial hidden nodes: {metadata['initial_hidden']} · test data seed: {metadata.get('test_data_seed', 'unknown')}"
        figure.suptitle(title, fontsize=13)
        labels = []
        for index, method in enumerate(m for m in METHODS if any(row["method"] == m for row in current)):
            values = [100 * row["test_accuracy"] for row in sorted(current, key=lambda row: row["seed"])
                      if row["method"] == method]
            label, color = METHODS[method]
            scores.scatter(values, index + np.linspace(-0.12, 0.12, len(values)), color=color, s=27, alpha=0.8)
            middle = float(np.median(values))
            scores.scatter(middle, index, marker="|", color="black", s=350, linewidths=2.5, zorder=4)
            labels.append(f"{label}\nn={len(values)}, median {middle:.2f}%")
        scores.set_yticks(range(len(labels)), labels)
        scores.tick_params(axis="y", labelsize=9)
        scores.invert_yaxis()
        scores.set_title("Test accuracy across evolution seeds", fontsize=11)
        scores.set_xlabel("Accuracy (%)")
        values = [100 * row["test_accuracy"] for row in current]
        scores.set_xlim(max(0, min(values) - 5), min(100, max(values) + 5))

        for index, item in enumerate(effects):
            gains = item["gains"]
            color = METHODS[item["method"]][1]
            differences.scatter(gains, index + np.linspace(-0.12, 0.12, len(gains)), color=color, s=27, alpha=0.7)
            if item["ci_low_pp"] is not None:
                differences.hlines(index, item["ci_low_pp"], item["ci_high_pp"], color="black", linewidth=2)
            differences.scatter(item["median_gain_pp"], index, marker="|", color="black", s=300, linewidths=2.5, zorder=4)
        if effects:
            differences.set_yticks(range(len(effects)),
                                  [f"{item['label']}\nmedian {item['median_gain_pp']:+.2f} pp, "
                                   f"W/T/L {item['wins']}/{item['ties']}/{item['losses']}" for item in effects])
        else:
            differences.set_yticks([])
            differences.text(0.5, 0.5, "No paired seeds available", transform=differences.transAxes, ha="center")
        differences.invert_yaxis()
        differences.tick_params(axis="y", labelsize=9)
        differences.axvline(0, color="#777777", linestyle="--", linewidth=1)
        differences.set_title("Paired accuracy gains (positive favors Waltz)", fontsize=11)
        differences.set_xlabel("Accuracy difference (percentage points)")
        for axes in (scores, differences):
            axes.spines[["top", "right"]].set_visible(False)
            axes.set_axisbelow(True)
            axes.grid(axis="x", alpha=0.2)
            axes.margins(y=0.15)
        footnote = ("Dots: individual seeds. Black ticks: medians. Horizontal lines: 95% bootstrap CIs for paired medians (n ≥ 2).\n"
                    "10,000 resamples, RNG seed 0; one fixed dataset. Paired seeds only. W/T/L: wins / ties / losses.")
        if "source_precision" in metadata:
            footnote += "\nSource accuracies rounded to 0.01 percentage points. Exploratory results used for tuning."
        figure.supxlabel(footnote, fontsize=8)
        for extension in ("png", "svg"):
            path = Path(f"{prefix}_g{generations}.{extension}")
            figure.savefig(path, dpi=300, bbox_inches="tight")
            paths.append(path)
    for item in summaries:
        interval = "not estimated" if item["ci_low_pp"] is None else f"[{item['ci_low_pp']:+.2f}, {item['ci_high_pp']:+.2f}]"
        print(f"{item['generations']} | {item['label']} | n={item['n']} | median {item['median_gain_pp']:+.2f} pp | "
              f"95% CI {interval} | W/T/L {item['wins']}/{item['ties']}/{item['losses']}")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    for path in save_plots(args.csv):
        print(f"Saved {path}")
