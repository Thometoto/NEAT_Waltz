"""Cyclic waltzers for neat-python 2.0.0."""

import random
from copy import copy, deepcopy
from math import inf, isfinite

import neat
from neat.innovation import InnovationTracker


def run_waltz(
    config, fitness, generations, *, islands=4, tempo=5, seed=None,
    on_visit=None, crossover=True, travel=True, waltzers=1, candidates_per_waltzer=1,
    tempo_schedule=None,
):
    """Return (best genome overall, best final waltzer by training fitness).

    fitness(genome, config) returns a finite score to maximize. The budget is
    generations * islands * config.pop_size, including the initial evaluation.
    on_visit receives the island, evaluated generation range, fitness before
    and after the visit, and the number of accepted candidates as keywords.
    Island indices start at 0. Candidate evaluations start at generation 1.
    Set crossover=False to mutate a clone of the waltzer on each visit.
    Set travel=False to keep each waltzer on its starting island.
    Set waltzers=0 for independent islands without candidate insertion;
    the second return value is then None and no visit events are emitted.
    Multiple waltzers start on distinct islands and move simultaneously.
    tempo_schedule=(15, 10, 5) overrides tempo: use each value for one complete
    tour, then keep the last value. A tour contains `islands` visits.
    Scheduled visit events also include the visit's tempo.
    The callback also receives waltzer_id and candidates_per_generation when
    either new parameter differs from 1. Existing single-waltzer calls retain
    their callback arguments and return values.
    """
    if islands < 2 or tempo < 1 or generations < 1 or config.pop_size < 2:
        raise ValueError("Use at least two islands and two genomes per island, with positive tempo and generations.")
    if type(waltzers) is not int or not 0 <= waltzers <= islands:
        raise ValueError("waltzers must be an integer between 0 and the island count")
    if type(candidates_per_waltzer) is not int or not 1 <= candidates_per_waltzer < config.pop_size:
        raise ValueError("candidates_per_waltzer must be a positive integer below the island population size")
    tempos = (tempo,) if tempo_schedule is None else tuple(tempo_schedule)
    if not tempos or any(type(value) is not int or value < 1 for value in tempos):
        raise ValueError("Tempos must be positive integers and the schedule must not be empty")

    config = copy(config)
    genome_config = config.genome_config
    original_node_indexer = genome_config.node_indexer
    original_tracker = genome_config.innovation_tracker
    random_state = random.getstate()

    try:
        genome_config.node_indexer = None
        random.seed(config.seed if seed is None else seed)
        config.seed = None
        populations = _create_populations(config, islands)
        best = None
        active_waltzers = []
        candidate_groups = []
        accepted = [0] * waltzers
        hosts = list(range(waltzers))
        visit_index = 0
        visit_start = 1
        current_tempo = tempos[0]
        visit_end = current_tempo

        for generation in range(generations):
            champions = [_evaluate_champion(island, fitness) for island in populations]
            champion = max(champions, key=lambda g: g.fitness)
            if best is None or champion.fitness > best.fitness:
                best = copy(champion)
            if generation == 0:
                active_waltzers = [copy(genome) for genome in champions[:waltzers]]
                arrival_fitness = [genome.fitness for genome in active_waltzers]
            else:
                for index, candidates in enumerate(candidate_groups):
                    candidate = max(candidates, key=lambda genome: genome.fitness)
                    if candidate.fitness > active_waltzers[index].fitness:
                        active_waltzers[index] = copy(candidate)
                        accepted[index] += 1

            final_generation = generation + 1 == generations
            if on_visit is not None and generation > 0 and (generation == visit_end or final_generation):
                # The candidate evaluated now was created in the previous generation.
                for index, waltzer in enumerate(active_waltzers):
                    event = dict(island=hosts[index], first_generation=visit_start,
                                 last_generation=generation, before=arrival_fitness[index],
                                 after=waltzer.fitness, accepted=accepted[index])
                    if waltzers != 1 or candidates_per_waltzer != 1:
                        event.update(waltzer_id=index, candidates_per_generation=candidates_per_waltzer)
                    if tempo_schedule is not None:
                        event["tempo"] = current_tempo
                    on_visit(**event)
                    arrival_fitness[index] = waltzer.fitness
                    accepted[index] = 0

            if final_generation:
                return best, max(active_waltzers, key=lambda genome: genome.fitness, default=None)

            if generation == visit_end:
                visit_index += 1
                current_tempo = tempos[min(visit_index // islands, len(tempos) - 1)]
                visit_start = generation + 1
                visit_end = generation + current_tempo
            offset = visit_index if travel else 0
            hosts = [(index + offset) % islands for index in range(waltzers)]
            candidate_groups = [[] for _ in range(waltzers)]
            for index, island in enumerate(populations):
                offspring = _reproduce(island, generation)
                if index in hosts:
                    waltzer_index = hosts.index(index)
                    local_champion = champions[index] if crossover else None
                    # Reserve slots before insertion so candidates cannot replace each other.
                    slots = sorted(offspring.values(), key=_replacement_priority)[:candidates_per_waltzer]
                    for slot in slots:
                        candidate = _insert_waltzer_candidate(
                            island, offspring, active_waltzers[waltzer_index], local_champion, slot=slot,
                        )
                        candidate_groups[waltzer_index].append(candidate)

                island.population = offspring
                island.species.speciate(config, offspring, generation)
    finally:
        genome_config.node_indexer = original_node_indexer
        genome_config.innovation_tracker = original_tracker
        random.setstate(random_state)


def _create_populations(config, island_count):
    """Create the islands with shared innovation IDs for crossover."""
    tracker = InnovationTracker()
    reproduction_type = config.reproduction_type

    def shared_reproduction(*args):
        reproduction = reproduction_type(*args)
        reproduction.innovation_tracker = tracker
        return reproduction

    config.reproduction_type = shared_reproduction
    populations = [neat.Population(config) for _ in range(island_count)]
    config.reproduction_type = reproduction_type
    return populations


def _evaluate_champion(island, fitness):
    """Score the current population and return its best genome."""
    genomes = island.population.values()
    for genome in genomes:
        genome.fitness = float(fitness(genome, island.config))
        if not isfinite(genome.fitness):
            raise ValueError("Fitness must be finite.")
    return max(genomes, key=lambda genome: genome.fitness)


def _reproduce(island, generation):
    """Run one standard NEAT reproduction step for an island."""
    config = island.config
    reproduction = island.reproduction
    offspring = reproduction.reproduce(config, island.species, config.pop_size, generation)
    if offspring:
        return offspring
    if not config.reset_on_extinction:
        raise neat.CompleteExtinctionException()
    return reproduction.create_new(config.genome_type, config.genome_config, config.pop_size)


def _replacement_priority(genome):
    return -inf if genome.fitness is None else genome.fitness


def _insert_waltzer_candidate(island, offspring, waltzer, local_champion, *, slot=None):
    """Mutate a clone or crossover child and give it one slot."""
    config = island.config
    key = next(island.reproduction.genome_indexer)
    if local_champion is None:
        candidate = deepcopy(waltzer)
        candidate.key = key
        candidate.fitness = None
    else:
        candidate = config.genome_type(key)
        candidate.configure_crossover(waltzer, local_champion, config.genome_config)
    candidate.mutate(config.genome_config)

    # Prefer replacing an unevaluated offspring over an elite.
    if slot is None:
        slot = min(offspring.values(), key=_replacement_priority)
    del offspring[slot.key]
    offspring[key] = candidate
    return candidate
