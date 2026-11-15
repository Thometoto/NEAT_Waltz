"""Run the frozen Circle and Spiral confirmation with three training processes."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3, help="Simultaneous training processes (default: 3)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/waltz_final_confirmation"))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    root = Path(__file__).resolve().parent
    output_dir = args.output_dir.resolve()
    outputs = [output_dir / task for task in ("circle", "spiral")]
    # Check both tasks before starting any expensive training.
    for output in outputs:
        if any(Path(f"{output}{suffix}").exists() for suffix in (".csv", ".json")):
            parser.error(f"Existing results at {output}; use a new --output-dir")
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    for task, output in zip(("circle", "spiral"), outputs):
        print(f"Starting {task.title()} confirmation: seeds 55–74, test seed 14, {args.workers} worker(s).", flush=True)
        subprocess.run([
            sys.executable, str(root / "compare_models.py"), "--task", task,
            "--generations", "2000", "--seeds", "55", "74", "--test-seed", "14",
            "--train-size", "512", "--mutation-power", "0.1", "--hidden", "4",
            "--waltzers", "4", "--candidates", "1", "--stage", "confirmation",
            "--workers", str(args.workers), "--output", str(output),
        ], cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
