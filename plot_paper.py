"""Build compact article figures from saved results, without any new training."""

import csv
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "docs/paper_figures"
COLORS = ("#0072B2", "#D55E00", "#009E73")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def paired_panel(ax, rows, summaries, target, baselines, labels, field, *, regression=False, exchange=False):
    """Recompute paired estimates and verify them against the original summary CSV."""
    a = {int(r["seed"]): float(r[field]) for r in rows if r["method"] == target}
    for i, baseline in enumerate(baselines):
        b = {int(r["seed"]): float(r[field]) for r in rows if r["method"] == baseline}
        if a.keys() != b.keys():
            raise ValueError("Compared methods must have exactly the same evolution seeds")
        gains = np.array([(b[s] - a[s]) if regression else (a[s] - b[s]) * 100 for s in sorted(a)])
        boot = np.median(np.random.default_rng(0).choice(gains, size=(10000, len(gains))), axis=1)
        low, high = np.quantile(boot, (0.025, 0.975))
        median = float(np.median(gains))
        summary = next(s for s in summaries if s["baseline"] == baseline and
                       (exchange or s["metric"] == "best_genome"))
        keys = ("median_gain_pp", "ci_low_pp", "ci_high_pp") if exchange else ("median_gain", "ci_low", "ci_high")
        if not np.allclose([median, low, high], [float(summary[k]) for k in keys], rtol=1e-12, atol=1e-12):
            raise ValueError("Recomputed plot statistics do not match the saved summary")
        ax.scatter(i + np.linspace(-0.13, 0.13, len(gains)), gains, s=15, alpha=0.65,
                   color="#777777", linewidths=0, zorder=2)
        ax.errorbar(i, median, yerr=[[median - low], [high - median]], marker="D", markersize=4,
                    color="#0072B2", linewidth=1.4, capsize=4, zorder=3)
    ax.axhline(0, color="#333333", linestyle="--", linewidth=0.9)
    ax.set(xticks=range(len(labels)), xticklabels=labels,
           ylabel="MSE reduction" if regression else "Accuracy gain (pp)")
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def save(figure, name):
    paths = []
    for extension in ("pdf", "png"):
        path = OUTPUT / f"{name}.{extension}"
        figure.savefig(path, dpi=220, metadata={"Creator": "NEAT Waltz plot_paper.py"} if extension == "pdf" else None)
        paths.append(path)
    return paths


def build_figures():
    names = ("tempo_effects", "exchange_effects", "sine_seed95")
    if any((OUTPUT / f"{name}.{ext}").exists() for name in names for ext in ("pdf", "png")):
        raise FileExistsError("Paper figures already exist; preserve this frozen figure set")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "font.family": "DejaVu Sans"})
    source_paths, paths = [], []
    figure = Figure(figsize=(6.2, 7.4), layout="constrained")
    for index, (task, ax) in enumerate(zip(("circle", "spiral", "sine"), figure.subplots(3, 1))):
        prefix = ROOT / "results/waltz_tempo_confirmation" / task
        csv_path, summary_path = Path(f"{prefix}.csv"), Path(f"{prefix}.summary.csv")
        source_paths.extend((csv_path, summary_path, Path(f"{prefix}.json")))
        rows, summaries = read_csv(csv_path), read_csv(summary_path)
        if len(rows) != 60 or len({(r["seed"], r["method"]) for r in rows}) != 60:
            raise ValueError("Final comparison requires sixty distinct runs per task")
        paired_panel(ax, rows, summaries, "accelerating", ("classic", "independent", "fixed5", "fixed15", "stationary"),
                     ("NEAT", "Islands", "Fixed 5", "Fixed 15", "Stationary"),
                     "test_mse" if task == "sine" else "test_accuracy", regression=task == "sine")
        ax.set_title(f"({chr(97 + index)}) {task.title()} | progressive Waltz minus baseline", loc="left")
        if task == "sine":
            ax.set_title("(c) Sine | baseline MSE minus progressive Waltz MSE", loc="left")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    figure.supxlabel("Grey dots: paired seeds (n = 10). Blue diamonds: medians; bars: pointwise 95% bootstrap intervals.\n"
                     "Positive values favor progressive Waltz. Outcomes use the best genome selected on training data.", fontsize=7)
    paths.extend(save(figure, "tempo_effects"))

    figure = Figure(figsize=(6.2, 5.3), layout="constrained")
    for index, (task, ax) in enumerate(zip(("circle", "spiral"), figure.subplots(2, 1))):
        prefix = ROOT / "results/waltz_exchange_confirmation" / task
        csv_path, summary_path = Path(f"{prefix}.csv"), Path(f"{prefix}.summary.csv")
        source_paths.extend((csv_path, summary_path, Path(f"{prefix}.json")))
        rows, summaries = read_csv(csv_path), read_csv(summary_path)
        if len(rows) != 100 or len({(r["seed"], r["method"]) for r in rows}) != 100:
            raise ValueError("Exchange confirmation requires one hundred distinct runs per task")
        paired_panel(ax, rows, summaries, "waltz_fixed_5", ("classic_neat", "independent", "waltz_stationary", "ring_migration"),
                     ("NEAT", "Islands", "Stationary", "Ring"), "test_accuracy", exchange=True)
        ax.set_title(f"({chr(97 + index)}) {task.title()} | traveling Waltz (fixed 5) minus baseline", loc="left")
    figure.supxlabel("Grey dots: paired seeds (n = 20). Blue diamonds: medians; bars: pointwise 95% bootstrap intervals.\n"
                     "Positive values favor traveling Waltz. This series uses separate datasets from the tempo comparison.", fontsize=7)
    paths.extend(save(figure, "exchange_effects"))

    figure = Figure(figsize=(6.2, 6.6), layout="constrained")
    methods = dict(classic="NEAT", independent="Independent islands", fixed5="Waltz, fixed 5",
                   fixed15="Waltz, fixed 15", accelerating="Waltz, progressive", stationary="Stationary")
    for index, (ax, (method, title)) in enumerate(zip(figure.subplots(3, 2).ravel(), methods.items())):
        source = ROOT / f"results/waltz_tempo_confirmation/sine_seed95_{method}.json"
        source_paths.append(source)
        data = json.loads(source.read_text())
        x = np.asarray(data["grid"])
        ax.plot(x, (1 + np.sin(2 * np.pi * x)) / 2, "--", color="black", linewidth=1, label="True function")
        ax.plot(x, data["best_curve"], color=COLORS[0], linewidth=1.3, label="Best training genome")
        if data["final_curve"] is not None:
            ax.plot(x, data["final_curve"], ":", color=COLORS[1], linewidth=1.3, label="Final waltzer")
        for edge in (0.25, 0.5, 0.75):
            ax.axvline(edge, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set(xlim=(0, 1), ylim=(-0.05, 1.05), xticks=(0, 0.5, 1), yticks=(0, 0.5, 1),
               xlabel="x", ylabel="y", title=f"({chr(97 + index)}) {title}\nTest MSE: {data['metrics']['test_mse']:.5f}")
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = figure.axes[2].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=3, fontsize=7)
    figure.suptitle("Sine | seed 95 (first confirmation seed)", fontsize=11)
    paths.extend(save(figure, "sine_seed95"))
    manifest = dict(purpose="Article figures only; no new training and no test-based example selection",
                    reference_seed=95, bootstrap_seed=0, bootstrap_resamples=10000,
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                    figure_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                    plotting_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    packages={name: version(name) for name in ("numpy", "matplotlib")})
    manifest_path = OUTPUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return [*paths, manifest_path]


if __name__ == "__main__":
    for output in build_figures():
        print(output)
