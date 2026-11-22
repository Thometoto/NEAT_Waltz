"""Replicate the Pendulum comparison on eight fresh banks with the pilot evaluator."""

import argparse
import fcntl
import hashlib
import json
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from math import isfinite
from multiprocessing import get_context
from pathlib import Path
from statistics import fmean, median
from time import perf_counter

import control_pendulum as pilot

METHODS = ("accelerating", "classic", "independent", "stationary")
ROOT = Path(__file__).resolve().parent


def verify(result, method, bank, seed, mode):
    m = result["metrics"]
    train, test = pilot.episode_seeds(bank, mode)
    generations = pilot.GENERATIONS[mode]
    with_child = method in ("accelerating", "stationary")
    expected = (method, bank, seed, mode, generations, 120 * generations,
                generations - 1 if with_child else 0, 120 * generations * len(train) * 200)
    actual = tuple(m[k] for k in ("method", "bank", "seed", "mode", "generations",
                                  "genome_evaluations", "candidate_evaluations", "training_steps"))
    if (actual != expected or (result["train_seeds"], result["test_seeds"]) != (train, test)
            or len(result["generation_history"]) != generations
            or len(result["best_episode_returns"]) != len(test)
            or any(not isfinite(v) for v in result["best_episode_returns"])
            or fmean(result["best_episode_returns"]) != m["best_test_return"]):
        raise ValueError(f"Inconsistent trial for {method}, bank {bank}, seed {seed}")
    return m


def summarize(out, rows):
    rows.sort(key=lambda r: (r["bank"], r["seed"], r["method"]))
    pilot.write_csv(out / "scores.csv", rows)
    scores = {(r["bank"], r["seed"], r["method"]): r["best_test_return"] for r in rows}
    pairs, banks, overall = [], [], []
    for comparator in METHODS[1:]:
        bank_medians = []
        for bank in sorted({r["bank"] for r in rows}):
            gains = []
            for seed in sorted({r["seed"] for r in rows if r["bank"] == bank}):
                gain = scores[bank, seed, "accelerating"] - scores[bank, seed, comparator]
                pairs.append(dict(comparator=comparator, bank=bank, seed=seed, gain=gain))
                gains.append(gain)
            bank_medians.append(median(gains))
            banks.append(dict(comparator=comparator, bank=bank, median_gain=median(gains),
                              wins=sum(g > 0 for g in gains), ties=gains.count(0),
                              losses=sum(g < 0 for g in gains)))
        overall.append(dict(comparator=comparator, median_bank_gain=median(bank_medians),
                            positive_banks=sum(g > 0 for g in bank_medians),
                            tied_banks=bank_medians.count(0), total_banks=len(bank_medians)))
    pilot.write_csv(out / "paired_differences.csv", pairs)
    pilot.write_csv(out / "banks.csv", banks)
    pilot.write_csv(out / "summary.csv", overall)


def run(out, args):
    mode = "smoke" if args.smoke else "pilot"
    banks = {90: [950]} if args.smoke else {b: list(range(400 + 5*i, 405 + 5*i))
                                             for i, b in enumerate(range(2, 10))}
    paths = (*pilot.pilot.SOURCES, "control_pendulum.py", "nonpersistent_control.py",
             "mutation_control.py", "confirm_pendulum.py", "docs/pendulum_confirmation.md")
    manifest = dict(study="pendulum_confirmation_v1", evaluation_mode=mode,
                    methods=list(METHODS), banks=[dict(bank=b, evolution_seeds=seeds,
                    train=pilot.episode_seeds(b, mode)[0], test=pilot.episode_seeds(b, mode)[1])
                    for b, seeds in banks.items()], generations=pilot.GENERATIONS[mode],
                    python=platform.python_version(), versions={p: version(p) for p in
                    ("numpy", "neat-python", "gymnasium", "matplotlib")},
                    source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths})
    protocol = out / "protocol.json"
    if protocol.exists():
        if not args.resume or json.loads(protocol.read_text()) != manifest:
            raise ValueError("Use a fresh directory or resume with the original protocol")
    elif any(p.name != ".lock" for p in out.iterdir()):
        raise ValueError("Output directory contains files without a protocol")
    else:
        pilot.save_json(protocol, manifest)
    jobs = [(method, b, s) for b, seeds in banks.items() for s in seeds for method in METHODS]
    rows, pending = [], {}
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        for method, bank, seed in jobs:
            path = out / f"{method}_bank{bank}_seed{seed}.json"
            if path.exists():
                rows.append(verify(json.loads(path.read_text()), method, bank, seed, mode))
            else:
                future = pool.submit(pilot.trial, method, bank, seed, mode)
                pending[future] = (path, method, bank, seed)
        for future in as_completed(pending):
            path, method, bank, seed = pending[future]
            result = future.result()
            rows.append(verify(result, method, bank, seed, mode))
            pilot.save_json(path, result)
            print(f"{len(rows)}/{len(jobs)} {path.stem}", flush=True)
    summarize(out, rows)
    if pending:
        pilot.save_json(out / "timing.json", dict(session_seconds=perf_counter()-start,
                        newly_completed=len(pending), workers=args.workers))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="Use a dedicated software-test bank")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    name = "pendulum_confirmation_smoke_v1" if args.smoke else "pendulum_confirmation_v1"
    out = args.output_dir or ROOT / "generated" / name
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another process is writing to this output directory")
        run(out, args)


if __name__ == "__main__":
    main()
