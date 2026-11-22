"""Final paired test of visit order. Run from the NEAT_Waltz repository root."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path
from statistics import fmean, median

METHODS = ("accelerating", "decelerating")
BANKS = range(40, 48)


def load_project():
    root = Path.cwd()
    if not (root / "control_pendulum.py").is_file():
        raise SystemExit("Start from the NEAT_Waltz directory containing control_pendulum.py")
    sys.path.insert(0, str(root))
    import control_pendulum
    return control_pendulum


def run_trial(method, bank, seed, mode):
    return load_project().trial(method, bank, seed, mode)


def check(raw, method, bank, seed, mode, pilot):
    train, test = pilot.episode_seeds(bank, mode)
    m = raw["metrics"]
    n = pilot.GENERATIONS[mode]
    expected = dict(method=method, bank=bank, seed=seed, mode=mode, generations=n,
                    genome_evaluations=n*120, candidate_evaluations=n-1,
                    training_steps=n*120*len(train)*200, heldout_steps=len(test)*400)
    if any(m[k] != v for k, v in expected.items()):
        raise ValueError(f"Unexpected budget or run identity: {method}, {bank}, {seed}")
    if (raw["train_seeds"], raw["test_seeds"]) != (train, test):
        raise ValueError("Episode seed mismatch")
    returns = raw["best_episode_returns"]
    if len(returns) != len(test) or not all(map(math.isfinite, returns)) or fmean(returns) != m["best_test_return"]:
        raise ValueError("Held-out returns do not match the primary score")
    if mode == "pilot":
        durations = [15]*8 + [10]*12 + [5]*24
        if method == "decelerating":
            durations.reverse()
        visits = raw["visits"]
        if ([v["last_generation"]-v["first_generation"]+1 for v in visits] != durations
                or [v["island"] for v in visits] != [i%4 for i in range(44)]):
            raise ValueError("Unexpected visit calendar")
    return m["best_test_return"]


def write_table(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def campaign(out, workers, pilot):
    seeds = {b: list(range(600+10*i, 610+10*i)) for i, b in enumerate(BANKS)}
    files = (*pilot.pilot.SOURCES, "control_pendulum.py", "mutation_control.py", "nonpersistent_control.py")
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    spec = dict(banks=[dict(bank=b, seeds=ss, train=pilot.episode_seeds(b,"pilot")[0],
                            test=pilot.episode_seeds(b,"pilot")[1]) for b,ss in seeds.items()],
                methods=list(METHODS), generations=361, steps_per_run=34656000,
                primary="Mean held-out return of training-selected global champion",
                contrast="accelerating minus decelerating, paired by bank and seed",
                decision="Directional support for acceleration if >=7 of 8 bank medians are strictly positive; report every difference and median of bank medians",
                sources={p: digest(p) for p in files}, launcher=digest(__file__),
                python=platform.python_version(), versions={p:version(p) for p in
                ("neat-python","gymnasium","numpy","matplotlib")})
    original = json.loads(Path("generated/pendulum_confirmation_v1/protocol.json").read_text())
    if (any(original["source_sha256"].get(p) != h for p,h in spec["sources"].items())
            or spec["versions"] != original["versions"] or spec["python"] != original["python"]):
        raise ValueError("Source or runtime differs from the completed Pendulum confirmation")
    path = out / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != spec:
            raise ValueError("Saved protocol differs from this launcher or runtime")
    else:
        if any(p.name != ".lock" for p in out.iterdir()):
            raise ValueError("Existing files without a protocol")
        pilot.save_json(path, spec)
    # Software check on separate starts, with no scientific runs included.
    for method in METHODS:
        check(pilot.trial(method, 91, 991, "smoke"), method, 91, 991, "smoke", pilot)
    scores, pending = {}, {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        for bank in BANKS:
            for seed in seeds[bank]:
                for method in METHODS:
                    key = (method,bank,seed)
                    path = out / f"{method}_bank{bank}_seed{seed}.json"
                    if path.exists():
                        scores[key] = check(json.loads(path.read_text()), *key, "pilot", pilot)
                    else:
                        pending[pool.submit(run_trial, *key, "pilot")] = (key,path)
        for future in as_completed(pending):
            key,path = pending[future]
            raw = future.result()
            scores[key] = check(raw, *key, "pilot", pilot)
            pilot.save_json(path,raw)
            print(f"{len(scores)}/160 {path.stem}",flush=True)
    pairs, banks = [], []
    for bank in BANKS:
        gains = []
        for seed in seeds[bank]:
            a, d = (scores[m,bank,seed] for m in METHODS)
            gains.append(a-d)
            pairs.append(dict(bank=bank,seed=seed,accelerating=a,decelerating=d,gain=a-d))
        banks.append(dict(bank=bank,median_gain=median(gains),wins=sum(x>0 for x in gains),
                          ties=gains.count(0),losses=sum(x<0 for x in gains)))
    write_table(out/"paired_differences.csv",pairs)
    write_table(out/"banks.csv",banks)
    pilot.save_json(out/"summary.json",dict(positive_banks=sum(r["median_gain"]>0 for r in banks),
                    total_banks=len(banks), median_bank_gain=median(r["median_gain"] for r in banks)))
    print(out/"summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=range(1,13), default=4)
    pilot = load_project()
    args = parser.parse_args()
    out = Path("generated/pendulum_final_tempo_v1")
    out.mkdir(parents=True,exist_ok=True)
    with (out/".lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This campaign is already running")
        campaign(out,args.workers,pilot)
