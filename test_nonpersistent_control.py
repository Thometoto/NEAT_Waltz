"""Check which two champions get picked at each visit."""

import json
import random
import unittest
from unittest.mock import patch

from nonpersistent_control import run_nonpersistent
from test_waltz_observations import config, genome_signature, xor_fitness


class NonpersistentTests(unittest.TestCase):
    def test_fresh_champions_every_round(self):
        from nonpersistent_control import _insert_waltzer_candidate
        selected = []
        calls = 0

        def score(genome, cfg):
            nonlocal calls
            calls += 1
            return float(genome.key)

        def insert(island, offspring, remote, local):
            selected.append((remote.key, remote.fitness, local.key, local.fitness))
            return _insert_waltzer_candidate(island, offspring, remote, local)

        with patch("nonpersistent_control._insert_waltzer_candidate", side_effect=insert):
            result = run_nonpersistent(config(), score, 15, seed=3, tempo_schedule=(2, 1))
        children, history = result["candidate_history"], result["generation_history"]
        self.assertEqual(calls, 15 * 4 * 6)
        self.assertEqual(len(children), 14)
        self.assertEqual([e["host"] for e in children], [0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3, 0, 1])
        self.assertTrue(all(e["lineages"] == [] and e["population_sizes"] == [6] * 4 for e in history))
        for child, parents in zip(children, selected):
            # The parents were chosen one generation before we scored their child.
            before = history[child["generation"] - 1]
            host = child["host"]
            donor = (host - 1) % 4
            self.assertEqual(child["remote_parent"]["island"], donor)
            self.assertEqual(child["local_parent"]["island"], host)
            self.assertEqual(parents[1], before["island_best_fitness"][donor])
            self.assertEqual(parents[3], before["island_best_fitness"][host])
            self.assertEqual(parents, (child["remote_parent"]["key"], child["remote_parent"]["fitness"],
                                       child["local_parent"]["key"], child["local_parent"]["fitness"]))
        expected = max([history[0]["island_best_fitness"][0]] + [e["candidate"]["fitness"] for e in children])
        self.assertEqual(result["stream_best"].fitness, expected)
        self.assertEqual(history[-1]["stream_best_fitness"], expected)
        json.dumps(dict(history=history, children=children, visits=result["visits"]))

    def test_reproducibility_and_state_restoration(self):
        cfg = config()
        state = random.getstate()
        original = cfg.genome_config.node_indexer, cfg.genome_config.innovation_tracker
        first = run_nonpersistent(cfg, xor_fitness, 10, seed=7, tempo_schedule=(2, 1))
        second = run_nonpersistent(cfg, xor_fitness, 10, seed=7, tempo_schedule=(2, 1))
        self.assertEqual(genome_signature(first["best"]), genome_signature(second["best"]))
        self.assertEqual(first["candidate_history"], second["candidate_history"])
        self.assertEqual(state, random.getstate())
        self.assertIs(cfg.genome_config.node_indexer, original[0])
        self.assertIs(cfg.genome_config.innovation_tracker, original[1])

    def test_initialization_only_and_failure(self):
        result = run_nonpersistent(config(), xor_fitness, 1, seed=4, tempo_schedule=(2,))
        self.assertEqual(result["candidate_history"], [])
        self.assertEqual(result["visits"], [])
        self.assertEqual(result["generation_history"][0]["genome_evaluations"], 24)
        state = random.getstate()
        with self.assertRaises(ValueError):
            run_nonpersistent(config(), lambda g, c: float("nan"), 2, seed=4, tempo_schedule=(2,))
        self.assertEqual(state, random.getstate())


if __name__ == "__main__":
    unittest.main()
