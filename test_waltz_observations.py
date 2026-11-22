"""Small runs to check when a waltzer moves and which child takes over.
Logging a visit should give the same run as leaving logging off.
"""

import json
import random
import unittest
from unittest.mock import patch

import neat

from diagnose_circle import create_config
from neat_waltz import _create_populations, run_waltz
from neat_waltz_domains import run_domains
from test_sine import sine_config, sine_regions
from test_xor import XOR, run_classic_neat
from waltz_tempo import balanced_schedule, three_phase_schedule


def config():
    result = create_config(2, mutation_power=0.1)
    result.pop_size = 6
    return result


def xor_fitness(genome, cfg):
    network = neat.nn.FeedForwardNetwork.create(genome, cfg)
    return -sum((network.activate(x)[0] - y) ** 2 for x, y in XOR) / 4


def genome_signature(genome):
    if genome is None:
        return None
    return (genome.key, genome.fitness,
            {k: vars(v) for k, v in genome.nodes.items()},
            {k: vars(v) for k, v in genome.connections.items()})


class ObservationTests(unittest.TestCase):
    def test_schedule_boundaries_partial_visit_and_budget(self):
        children, rounds, visits = [], [], []
        calls = 0

        def score(genome, cfg):
            nonlocal calls
            calls += 1
            return float(genome.key)

        best, final = run_waltz(
            config(), score, 10, islands=2, tempo_schedule=(2, 1), seed=3,
            waltzers=2, candidates_per_waltzer=2,
            on_candidate=lambda **e: children.append(e),
            on_generation=lambda **e: rounds.append(e), on_visit=lambda **e: visits.append(e),
        )
        self.assertEqual(calls, 10 * 2 * 6)
        self.assertEqual(len(rounds), 10)
        self.assertEqual(len(children), 9 * 2 * 2)
        self.assertEqual(rounds[-1]["genome_evaluations"], calls)
        self.assertEqual(rounds[-1]["candidate_evaluations"], len(children))
        self.assertTrue(all(e["population_sizes"] == [6, 6] for e in rounds))
        self.assertEqual([e["island"] for e in children if e["waltzer_id"] == 0][::2],
                         [0, 0, 1, 1, 0, 1, 0, 1, 0])
        previous = [r["lineages"] for r in rounds]
        for generation in range(1, 10):
            for index in range(2):
                group = [e for e in children if e["generation"] == generation and e["waltzer_id"] == index]
                before = group[0]["fitness_before"]
                maximum = max(e["candidate"]["fitness"] for e in group)
                self.assertEqual(sum(e["accepted"] for e in group), int(maximum > before))
                self.assertEqual(previous[generation][index]["fitness"], max(before, maximum))
                for event in group:
                    self.assertEqual(event["created_generation"], generation - 1)
                    self.assertEqual(event["local_parent"]["island"], event["island"])
                    for field in ("island", "key", "fitness"):
                        self.assertEqual(event["waltzer_parent"][field], previous[generation - 1][index][field])
                    self.assertEqual(event["fitness_gain"], event["candidate"]["fitness"] - before)
        for index in range(2):
            events = [e for e in visits if e["waltzer_id"] == index]
            self.assertEqual([g for e in events for g in range(e["first_generation"], e["last_generation"] + 1)],
                             list(range(1, 10)))
        self.assertGreaterEqual(best.fitness, final.fitness)
        json.dumps(dict(children=children, generations=rounds))

    def test_ties_reject_all_children_and_final_visit_can_be_short(self):
        children, visits = [], []
        _, final = run_waltz(config(), lambda g, c: 0.0, 4, islands=2, tempo=8, seed=4,
                             candidates_per_waltzer=2,
                             on_candidate=lambda **e: children.append(e),
                             on_visit=lambda **e: visits.append(e))
        self.assertEqual(len(children), 6)
        self.assertFalse(any(e["accepted"] for e in children))
        self.assertEqual(len(visits), 1)
        self.assertEqual((visits[0]["first_generation"], visits[0]["last_generation"]), (1, 3))
        self.assertTrue(all(e["waltzer_parent"]["key"] == final.key for e in children))

    def test_initialization_only_and_independent_islands(self):
        for generations, waltzers in ((1, 1), (5, 0)):
            children, rounds, visits = [], [], []
            _, final = run_waltz(config(), xor_fitness, generations, islands=2, seed=4,
                                 waltzers=waltzers, on_candidate=lambda **e: children.append(e),
                                 on_generation=lambda **e: rounds.append(e),
                                 on_visit=lambda **e: visits.append(e))
            self.assertEqual(children, [])
            self.assertEqual(visits, [])
            self.assertEqual(rounds[-1]["genome_evaluations"], generations * 12)
            self.assertEqual(final is None, waltzers == 0)

    def test_observers_preserve_genomes_rng_and_configuration(self):
        for travel, crossover in ((True, True), (False, True), (True, False)):
            cfg = config()
            state = random.getstate()
            tracker, node_indexer = cfg.genome_config.innovation_tracker, cfg.genome_config.node_indexer
            kwargs = dict(islands=2, seed=7, tempo_schedule=(2, 1), travel=travel,
                          crossover=crossover, waltzers=2, candidates_per_waltzer=2)
            reference = run_waltz(cfg, xor_fitness, 12, **kwargs)
            observed = run_waltz(cfg, xor_fitness, 12, on_candidate=lambda **e: None,
                                 on_generation=lambda **e: None, **kwargs)
            self.assertEqual([genome_signature(g) for g in reference],
                             [genome_signature(g) for g in observed])
            self.assertEqual(state, random.getstate())
            self.assertIs(cfg.genome_config.innovation_tracker, tracker)
            self.assertIs(cfg.genome_config.node_indexer, node_indexer)

    def test_invalid_counts_and_nonfinite_fitness(self):
        for options in (dict(islands=2.5), dict(generations=2.5), dict(tempo=True)):
            kwargs = dict(generations=3, islands=2)
            kwargs.update(options)
            with self.assertRaises(ValueError):
                run_waltz(config(), xor_fitness, **kwargs)
        cfg, state = config(), random.getstate()
        with self.assertRaises(ValueError):
            run_waltz(cfg, lambda g, c: float("nan"), 3, islands=2, seed=7)
        self.assertEqual(state, random.getstate())

    def test_shared_initial_homology_and_distinct_population_objects(self):
        cfg = config()
        state = random.getstate()
        try:
            populations = _create_populations(cfg, 2)
            self.assertIs(populations[0].reproduction.innovation_tracker,
                          populations[1].reproduction.innovation_tracker)
            a, b = [next(iter(p.population.values())) for p in populations]
            self.assertIsNot(a, b)
            self.assertEqual(set(a.nodes), set(b.nodes))
            self.assertEqual({k: g.innovation for k, g in a.connections.items()},
                             {k: g.innovation for k, g in b.connections.items()})
        finally:
            random.setstate(state)

    def test_population_size_violation_is_detected(self):
        from neat_waltz import _reproduce

        def lose_one(island, generation):
            offspring = _reproduce(island, generation)
            offspring.pop(next(iter(offspring)))
            return offspring

        with patch("neat_waltz._reproduce", side_effect=lose_one):
            with self.assertRaisesRegex(RuntimeError, "population size"):
                run_waltz(config(), xor_fitness, 3, islands=2, seed=3, waltzers=0)

    def test_domain_observations_use_global_acceptance_and_local_selection(self):
        regions = sine_regions(3, 11)
        for mode in ("classic", "independent", "waltz", "stationary", "mutation", "waltz_no_mutation"):
            cfg = sine_config(2)
            cfg.pop_size = 6
            children, rounds = [], []
            options = dict(mode=mode, seed=7, tempo_schedule=(2, 1), waltzers=2)
            reference = run_domains(cfg, regions, 13, **options)
            observed = run_domains(cfg, regions, 13, on_candidate=lambda **e: children.append(e),
                                   on_generation=lambda **e: rounds.append(e), **options)
            self.assertEqual(genome_signature(reference["best"]), genome_signature(observed["best"]))
            self.assertEqual([genome_signature(g) for g in reference["lineages"]],
                             [genome_signature(g) for g in observed["lineages"]])
            self.assertEqual(reference["visits"], observed["visits"])
            self.assertEqual(observed["genome_evaluations"], 13 * 24)
            self.assertEqual(observed["training_predictions"], 13 * 24 * 12)
            self.assertEqual(observed["candidate_evaluations"], len(children))
            self.assertEqual(len(rounds), 13)
            for event in children:
                self.assertEqual(event["accepted"], event["fitness_gain"] > 0)
                self.assertEqual(event["waltzer_parent"]["fitness"], event["fitness_before"])
                self.assertEqual(event["local_parent"] is None, mode == "mutation")
            self.assertTrue(all("island_best_global_fitness" in r for r in rounds))
            json.dumps(dict(children=children, generations=rounds))

    def test_confirmation_schedule_spends_full_tours_in_each_phase(self):
        schedule, phases = balanced_schedule(2000)
        self.assertEqual([(p["first_generation"], p["last_generation"], p["tempo"]) for p in phases],
                         [(1, 660, 15), (661, 1340, 10), (1341, 1999, 5)])
        self.assertEqual(schedule, (15,) * 11 + (10,) * 17 + (5,))

    def test_classic_training_trace_and_budget(self):
        rounds = []
        expected, _ = run_classic_neat(config(), xor_fitness, 7, 9)
        best, _ = run_classic_neat(config(), xor_fitness, 7, 9,
                                   on_generation=lambda **e: rounds.append(e))
        self.assertEqual(genome_signature(expected), genome_signature(best))
        self.assertEqual(len(rounds), 7)
        self.assertEqual(rounds[-1]["genome_evaluations"], 7 * 24)
        self.assertEqual(rounds[-1]["best_fitness"], best.fitness)
        self.assertTrue(all(e["lineages"] == [] for e in rounds))

    def test_pilot_phases_and_movement_counts(self):
        from pilot_tempo import visit_options

        for method, expected_moves in (("accelerating", 43), ("decelerating", 43),
                                       ("constant", 35), ("stationary", 0)):
            children, visits = [], []
            run_waltz(config(), lambda g, c: 0.0, 361, seed=3,
                       **visit_options(method, 361),
                       on_candidate=lambda **e: children.append(e),
                       on_visit=lambda **e: visits.append(e))
            self.assertEqual(len(children), 360)
            moves = sum(a["island"] != b["island"] for a, b in zip(visits, visits[1:]))
            self.assertEqual(moves, expected_moves)
            if method != "constant":
                tempos = (5, 10, 15) if method == "decelerating" else (15, 10, 5)
                self.assertEqual([e["tempo"] for e in children],
                                 [tempo for tempo in tempos for _ in range(120)])

    def test_three_phase_schedule_rejects_incomplete_tours(self):
        for generations in (1, 121, 360, 362):
            with self.assertRaises(ValueError):
                three_phase_schedule(generations)
        _, phases = three_phase_schedule(361)
        self.assertEqual([(p["first_generation"], p["last_generation"]) for p in phases],
                         [(1, 120), (121, 240), (241, 360)])


if __name__ == "__main__":
    unittest.main()
