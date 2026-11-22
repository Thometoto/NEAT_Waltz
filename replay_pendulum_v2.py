"""Replay a saved Pendulum controller and write the animation to an HTML file."""

import argparse
import json
import math
from pathlib import Path

import gymnasium as gym
import neat
import numpy as np
from neat.activations import sigmoid_activation
from neat.aggregations import sum_aggregation

from pendulum_waltz_v2 import UPRIGHT_COS, action


def restore(record):
    nodes = []
    for n in record["nodes"]:
        if (n["activation"], n["aggregation"]) != ("sigmoid_activation", "sum_aggregation"):
            raise ValueError("Unsupported saved network")
        nodes.append((n["id"], sigmoid_activation, sum_aggregation,
                      n["bias"], n["response"], n["links"]))
    return neat.nn.FeedForwardNetwork(record["inputs"], record["outputs"], nodes)


def replay(env, network, seed, energy):
    obs, _ = env.reset(seed=seed)
    frames = [[float(obs[0]), float(obs[1]), 0., 0.]]
    stable = 0
    total = 0.
    for step in range(200):
        torque = np.float32(action(network, obs, energy))
        obs, reward, terminated, truncated, _ = env.step(np.array([torque], dtype=np.float32))
        if terminated or truncated != (step == 199):
            raise RuntimeError("Unexpected episode length")
        total += float(reward)
        if step >= 150 and obs[0] >= UPRIGHT_COS and abs(obs[2]) <= 1:
            stable += 1
        frames.append([float(obs[0]), float(obs[1]), float(torque), total])
    return dict(seed=seed, frames=frames, stable=stable)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    row = json.loads(args.record.read_text())
    network = restore(row["network"])
    with gym.make("Pendulum-v1", g=10.) as env:
        episodes = [replay(env, network, s, row["method"].endswith("_energy")) for s in row["test_seeds"]]
    for episode, expected, success in zip(episodes, row["episode_returns"], row["episode_success"], strict=True):
        if not math.isclose(episode["frames"][-1][-1], expected, abs_tol=1e-8, rel_tol=0) or (episode["stable"] >= 45) != success:
            raise ValueError("Saved network did not reproduce the recorded episode")
    template = Path(__file__).with_name("docs").joinpath("pendulum_replay.html").read_text()
    payload = dict(method=row["method"], bank=row["bank"], seed=row["seed"], episodes=episodes)
    args.output.write_text(template.replace("EPISODE_DATA", json.dumps(payload,allow_nan=False,separators=(",",":"))))
    print(args.output)


if __name__ == "__main__":
    main()
