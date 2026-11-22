"""Prespecified visit durations for the last Waltz experiment."""


def three_phase_schedule(generations, tempos=(15, 10, 5), *, islands=4):
    """Return equal-duration phases containing complete tours, plus their bounds.

    Use 361, 721, ... generations for (15, 10, 5) on four islands.
    Reversing tempos preserves phase lengths and the total number of visits.
    Unlike balanced_schedule, no rounding or truncated final visit is needed.
    """
    tempos = tuple(tempos)
    if (type(generations) is not int or generations < 2 or type(islands) is not int
            or islands < 2 or len(tempos) != 3
            or any(type(t) is not int or t < 1 for t in tempos)
            or (generations - 1) % 3):
        raise ValueError("Use three positive tempos and three equal integer phases")
    rounds = (generations - 1) // 3
    if any(rounds % (islands * t) for t in tempos):
        raise ValueError("Each phase must contain complete tours; use 361, 721, ... generations for 15/10/5")
    schedule, phases = [], []
    for index, tempo in enumerate(tempos):
        schedule.extend([tempo] * (rounds // (islands * tempo)))
        phases.append(dict(first_generation=index * rounds + 1,
                           last_generation=(index + 1) * rounds, tempo=tempo))
    return tuple(schedule), phases


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
