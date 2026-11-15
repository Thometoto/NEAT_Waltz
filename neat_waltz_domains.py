"""Domain-specialized islands with a common training criterion for traveling lineages."""

import random
from copy import deepcopy
from math import isfinite

import neat

from neat_migration import _insert_migrant
from neat_waltz import _create_populations, _insert_waltzer_candidate, _replacement_priority, _reproduce


MODES = ("classic", "independent", "waltz", "stationary", "mutation", "migration")
CROSSOVER_MODES = ("classic", "independent", "waltz", "stationary", "waltz_no_mutation",
                   "stationary_no_mutation", "mutation", "migration")


def insert_crossover_only(island, offspring, waltzer, champion):
    """Insert one crossover child; ordinary population mutation is unaffected."""
    config = island.config
    key = next(island.reproduction.genome_indexer)
    candidate = config.genome_type(key)
    candidate.configure_crossover(waltzer, champion, config.genome_config)
    slot = min(offspring.values(), key=_replacement_priority)
    del offspring[slot.key]
    offspring[key] = candidate
    return candidate


def region_errors(network, regions):
    """Compute MSE separately on each nonempty region; use no test data here."""
    return tuple(sum((network.activate(inputs)[0] - target) ** 2 for inputs, target in region) / len(region)
                 for region in regions)


def global_copy(genome):
    """Archive a genome with its comparable, global training fitness."""
    archived = deepcopy(genome)
    archived.fitness = -sum(archived.region_mse) / len(archived.region_mse)
    return archived


def run_domains(config, regions, generations, *, mode="waltz", seed=0, tempo=5, waltzers=1,
                tempo_schedule=None):
    """Return the best global-training genome, final lineages and training-only traces.

    Four equal-size regions. Every evaluated genome predicts all training points,
    exactly once, in every mode. Island reproduction uses ONLY its own region's
    MSE; global scores select archived outputs and accept lineage candidates.
    Crossover compares both parents on global training fitness, never local scores
    from different domains. No test points enter this function.
    Optional tempo_schedule uses each duration for one complete four-visit tour,
    then retains the last duration, as in run_waltz. Fixed-tempo calls are unchanged.
    """
    if mode not in CROSSOVER_MODES or len(regions) != 4 or not regions[0] or len({len(r) for r in regions}) != 1:
        raise ValueError("Use a supported mode and four nonempty equally sized training regions")
    if (type(generations) is not int or generations < 1 or type(tempo) is not int or tempo < 1
            or type(waltzers) is not int or not 1 <= waltzers <= 4 or config.pop_size < 2):
        raise ValueError("Use positive integer generations/tempo, 1–4 waltzers and at least two genomes per island")
    tempos = (tempo,) if tempo_schedule is None else tuple(tempo_schedule)
    if not tempos or any(type(t) is not int or t < 1 for t in tempos):
        raise ValueError("Tempos must be positive integers and the schedule must not be empty")
    config = deepcopy(config)
    state = random.getstate()
    try:
        random.seed(seed)
        config.seed = None
        config.genome_config.node_indexer = None
        island_size = config.pop_size
        if mode == "classic":
            config.pop_size *= 4
        populations = _create_populations(config, 1 if mode == "classic" else 4)
        lineage_count = waltzers if mode in ("waltz", "stationary", "mutation", "waltz_no_mutation", "stationary_no_mutation") else 0
        starts = [(seed + j) % 4 for j in range(lineage_count)]
        hosts = starts[:]
        lineages, pending, history, snapshots = [], [], [], []
        visits, accepted = [], [0] * lineage_count
        arrivals = []
        best, evaluations, predictions = None, 0, 0
        candidate_evaluations, migrant_evaluations = 0, 0
        snapshot_rounds = {0, 4 * tempo, 8 * tempo, generations - 1}
        visit_index, visit_start = 0, 1
        current_tempo = visit_end = tempos[0]

        for generation in range(generations):
            champions = []
            for host, island in enumerate(populations):
                for genome in island.population.values():
                    network = neat.nn.FeedForwardNetwork.create(genome, config)
                    genome.region_mse = region_errors(network, regions)
                    evaluations += 1
                    predictions += sum(len(region) for region in regions)
                    global_fitness = -sum(genome.region_mse) / 4
                    if not isfinite(global_fitness):
                        raise ValueError("Training fitness must be finite")
                    genome.fitness = global_fitness if mode == "classic" else -genome.region_mse[host]
                    if best is None or global_fitness > best.fitness:
                        best = global_copy(genome)
                champions.append(max(island.population.values(), key=lambda g: g.fitness))

            if generation == 0:
                lineages = [global_copy(champions[host]) for host in starts]
                arrivals = [list(genome.region_mse) for genome in lineages]
            else:
                for j, candidate in enumerate(pending):
                    candidate_evaluations += 1
                    candidate = global_copy(candidate)
                    if candidate.fitness > lineages[j].fitness:
                        lineages[j] = candidate
                        accepted[j] += 1

            for j, genome in enumerate(lineages):
                history.append(dict(generation=generation, waltzer=j, host=hosts[j],
                                    global_train_mse=-genome.fitness, region_mse=list(genome.region_mse)))
            if lineages and generation in snapshot_rounds:
                snapshots.append((generation, deepcopy(lineages[0])))
            final = generation + 1 == generations
            if generation > 0 and (generation == visit_end or final):
                for j, genome in enumerate(lineages):
                    event = dict(waltzer=j, host=hosts[j],
                                       first_generation=visit_start,
                                       last_generation=generation, accepted=accepted[j],
                                       region_mse_before=arrivals[j], region_mse_after=list(genome.region_mse))
                    if tempo_schedule is not None:
                        event["tempo"] = current_tempo
                    visits.append(event)
                    arrivals[j] = list(genome.region_mse)
                    accepted[j] = 0
            if final:
                break

            # Candidate evaluated at g=1..tempo stays on the initial host.
            if generation == visit_end:
                visit_index += 1
                current_tempo = tempos[min(visit_index // 4, len(tempos) - 1)]
                visit_start = generation + 1
                visit_end = generation + current_tempo
            offset = visit_index if mode not in ("stationary", "stationary_no_mutation") else 0
            hosts = [(start + offset) % 4 for start in starts]
            pending = []
            migrate = mode == "migration" and (generation + 1) % tempo == 0
            donors = deepcopy(champions) if migrate else []
            for host, island in enumerate(populations):
                offspring = _reproduce(island, generation)
                if host in hosts:
                    j = hosts.index(host)
                    # Both parent fitness values use the same global training data.
                    local = global_copy(champions[host]) if mode != "mutation" else None
                    insert = insert_crossover_only if mode in ("waltz_no_mutation", "stationary_no_mutation") else _insert_waltzer_candidate
                    pending.append((j, insert(island, offspring, lineages[j], local)))
                if migrate:
                    _insert_migrant(island, offspring, donors[(host - 1) % 4])
                    migrant_evaluations += 1
                island.population = offspring
                island.species.speciate(config, offspring, generation)
            pending = [genome for _, genome in sorted(pending)]

        expected = generations * 4 * island_size
        if evaluations != expected or predictions != expected * sum(map(len, regions)):
            raise RuntimeError("Training evaluation/prediction budget mismatch")
        if candidate_evaluations != (generations - 1) * lineage_count:
            raise RuntimeError("Lineage candidate accounting mismatch")
        return dict(best=best, lineages=lineages, history=history, visits=visits, snapshots=snapshots,
                    genome_evaluations=evaluations, training_predictions=predictions,
                    candidate_evaluations=candidate_evaluations, migrant_evaluations=migrant_evaluations)
    finally:
        random.setstate(state)
