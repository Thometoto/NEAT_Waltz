"""Run two-arm spiral classification with python test_spiral.py."""

from math import cos, pi, sin, tau
from random import Random

from test_xor import run_classification


def spiral_examples(points_per_class, seed):
    """Sample two interleaved spiral arms, with one turn per arm."""
    rng = Random(seed)
    examples = []
    for label in (0, 1):
        for _ in range(points_per_class):
            position = rng.random()
            radius = 0.15 + 0.85 * position
            angle = tau * position + label * pi
            examples.append(((radius * cos(angle), radius * sin(angle)), label))
    rng.shuffle(examples)
    return examples


if __name__ == "__main__":
    train = spiral_examples(64, seed=10)
    run_classification("Spiral", train, generations=250,
                       make_test=lambda seed: spiral_examples(128, seed), make_train=spiral_examples)
