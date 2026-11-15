"""Run inside/outside circle classification with python test_circle.py."""

from random import Random

from test_xor import run_classification


def circle_examples(points_per_class, seed):
    """Sample balanced points in the square [-1, 1]² around a radius-0.8 circle."""
    rng = Random(seed)
    inside, outside = [], []
    while min(len(inside), len(outside)) < points_per_class:
        x, y = rng.uniform(-1, 1), rng.uniform(-1, 1)
        label = int(x * x + y * y <= 0.8 ** 2)
        group = inside if label else outside
        if len(group) < points_per_class:
            group.append(((x, y), label))
    examples = inside + outside
    rng.shuffle(examples)
    return examples


if __name__ == "__main__":
    train = circle_examples(64, seed=10)
    run_classification("Circle", train, generations=150,
                       make_test=lambda seed: circle_examples(128, seed), make_train=circle_examples)
