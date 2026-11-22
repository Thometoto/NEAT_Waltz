"""Replay the saved Pendulum champions and check their scores."""
import csv
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path
from statistics import fmean, median
from zipfile import ZIP_DEFLATED, ZipFile

import gymnasium as gym

from replay_pendulum_v2 import replay, restore

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "generated" / "pendulum_waltz_v2_evaluation"
RESULTS = ROOT / "results" / "pendulum_waltz_v2"


def check(condition, message):
    if not condition:
        raise ValueError(message)


def same_score(actual, expected):
    return math.isclose(actual, expected, rel_tol=0, abs_tol=1e-8)


def write_csv(name, rows):
    with (RESULTS / name).open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def verify_trial(env, method, bank, seed):
    path = RUNS / f"{method}_b{bank}_s{seed}.json"
    row = json.loads(path.read_text())
    check((row["method"], row["bank"], row["seed"]) == (method, bank, seed), path)
    check((row["generations"], row["evaluations"], row["training_steps"]) ==
          (73, 8760, 7008000), f"Budget mismatch in {path}")
    train = list(range(120000 + 1000 * bank, 120004 + 1000 * bank))
    test = list(range(220000 + 1000 * bank, 220032 + 1000 * bank))
    check((row["train_seeds"], row["test_seeds"]) == (train, test), path)
    network = restore(row["network"])
    energy = method.endswith("_energy")
    scores, flags = [], []
    for start, expected, success in zip(test, row["episode_returns"], row["episode_success"], strict=True):
        episode = replay(env, network, start, energy)
        score = episode["frames"][-1][-1]
        stable = episode["stable"] >= 45
        check(same_score(score, expected) and stable == success, f"Replay mismatch in {path} start {start}")
        scores.append(score)
        flags.append(stable)
    train_score = fmean(replay(env, network, s, energy)["frames"][-1][-1] for s in train)
    check(same_score(train_score, row["train_return"]), f"Training score mismatch in {path}")
    check(same_score(fmean(scores), row["mean_return"]) and sum(flags) == row["successes"], path)
    return dict(method=method, bank=bank, seed=seed, mean_return=fmean(scores), successes=sum(flags))


def main():
    protocol = json.loads((RUNS / "protocol.json").read_text())
    for name, digest in protocol["source_sha256"].items():
        check(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f"Changed source: {name}")
    for package, expected in protocol["versions"].items():
        check(version(package) == expected, f"Changed version: {package}")
    methods, pairs = protocol["methods"], protocol["pairs"]
    check(len(set(methods)) == 6 and len(set(map(tuple, pairs))) == 9, "Unexpected protocol")
    check(len(list(RUNS.glob("*_b*_s*.json"))) == 54, "Expected 54 completed trials")
    with gym.make("Pendulum-v1", g=10.) as env:
        rows = {(method, bank, seed): verify_trial(env, method, bank, seed)
                for bank, seed in pairs for method in methods}
    RESULTS.mkdir(exist_ok=True, parents=True)
    write_csv("verified_scores.csv", list(rows.values()))
    summary, comparisons = [], []
    for method in methods:
        group = [rows[method, bank, seed] for bank, seed in pairs]
        summary.append(dict(method=method, controllers=len(group),
                            solved=sum(r["successes"] >= 29 for r in group),
                            successful_episodes=sum(r["successes"] for r in group),
                            episodes=32 * len(group),
                            median_return=median(r["mean_return"] for r in group)))
        if method == "waltz2_energy":
            continue
        for bank in sorted({b for b, seed in pairs}):
            # Compare champions trained with the same seed on the same starts.
            deltas = [rows["waltz2_energy", b, seed]["mean_return"] - rows[method, b, seed]["mean_return"]
                      for b, seed in pairs if b == bank]
            comparisons.append(dict(reference=method, bank=bank, pairs=len(deltas),
                                    median_delta=median(deltas), wins=sum(d > 0 for d in deltas),
                                    ties=deltas.count(0), losses=sum(d < 0 for d in deltas)))
    write_csv("summary.csv", summary)
    write_csv("paired_by_bank.csv", comparisons)
    with ZipFile(RESULTS / "verified_controllers.zip", "w", compression=ZIP_DEFLATED) as archive:
        archive.write(RUNS / "protocol.json", "protocol.json")
        for method, bank, seed in rows:
            filename = f"{method}_b{bank}_s{seed}.json"
            archive.write(RUNS / filename, filename)
    print("Verified 54 controllers, 216 training episodes and 1,728 held-out episodes.")
    print((RESULTS / "summary.csv").read_text())
    print((RESULTS / "paired_by_bank.csv").read_text())


if __name__ == "__main__":
    main()
