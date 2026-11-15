"""Prespecified visit durations for the last Waltz experiment."""


def balanced_schedule(generations):
    """Allocate about one third of candidate rounds to each of 15, 10 and 5.

    Generation 0 initializes the lineages. Transitions follow whole tours
    (four visits); the final visit may be truncated by the evaluation budget.
    At 2000 generations: 1–660 at 15, 661–1340 at 10, 1341–1999 at 5.
    """
    if type(generations) is not int or generations < 102:
        raise ValueError("Use at least 102 generations to include all three tempo phases")
    rounds = generations - 1
    first_tours = max(1, int(round(rounds / 3 / 60)))
    second_tours = max(1, int(round((2 * rounds / 3 - first_tours * 60) / 40)))
    first, second = first_tours * 60, second_tours * 40
    if first + second >= rounds:
        raise ValueError("The budget must leave candidate rounds for the final tempo")
    schedule = (15,) * first_tours + (10,) * second_tours + (5,)
    phases = [dict(first_generation=1, last_generation=first, tempo=15),
              dict(first_generation=first + 1, last_generation=first + second, tempo=10),
              dict(first_generation=first + second + 1, last_generation=rounds, tempo=5)]
    return schedule, phases
