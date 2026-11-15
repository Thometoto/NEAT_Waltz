"""Periodic champion-copy migration in a synchronous ring of NEAT islands."""

import random
from copy import copy, deepcopy

from neat_waltz import _create_populations, _evaluate_champion, _replacement_priority, _reproduce


def _insert_migrant(island, offspring, donor):
    """Copy a donor into one existing slot; never mutate or move the donor."""
    migrant = deepcopy(donor)
    migrant.key = next(island.reproduction.genome_indexer)
    migrant.fitness = None
    slot = min(offspring.values(), key=_replacement_priority)
    del offspring[slot.key]
    offspring[migrant.key] = migrant
    return migrant


def run_ring_migration(config, fitness, generations, *, islands=4, interval=5, seed=None,
                       on_migration=None):
    """Return the best training genome seen, using generations * islands * P evaluations.

    One current champion per island is copied to the next island in the ring.
    Migrants are evaluated at generations interval, 2*interval, ... below G.
    All donors come from the same evaluated generation before any insertion.
    Ordinary NEAT reproduction continues between migration events. Unlike Waltz,
    there is no persistent external lineage, forced crossover or acceptance gate.
    """
    if (type(islands) is not int or islands < 2 or type(interval) is not int or interval < 1
            or type(generations) is not int or generations < 1 or config.pop_size < 2):
        raise ValueError("Use at least two islands/two genomes and positive integer generations/interval")
    config = copy(config)
    genome_config = config.genome_config
    original_indexer = genome_config.node_indexer
    original_tracker = genome_config.innovation_tracker
    random_state = random.getstate()
    try:
        genome_config.node_indexer = None
        random.seed(config.seed if seed is None else seed)
        config.seed = None
        populations = _create_populations(config, islands)
        best = None
        for generation in range(generations):
            champions = [_evaluate_champion(island, fitness) for island in populations]
            champion = max(champions, key=lambda genome: genome.fitness)
            if best is None or champion.fitness > best.fitness:
                best = copy(champion)
            if generation + 1 == generations:
                return best

            # Snapshot every donor before any host reproduces or receives a copy.
            migrate = (generation + 1) % interval == 0
            donors = [deepcopy(champion) for champion in champions] if migrate else []
            for host, island in enumerate(populations):
                offspring = _reproduce(island, generation)
                if migrate:
                    source = (host - 1) % islands
                    _insert_migrant(island, offspring, donors[source])
                    if on_migration is not None:
                        on_migration(source=source, destination=host,
                                     evaluated_generation=generation + 1)
                island.population = offspring
                island.species.speciate(config, offspring, generation)
    finally:
        genome_config.node_indexer = original_indexer
        genome_config.innovation_tracker = original_tracker
        random.setstate(random_state)
