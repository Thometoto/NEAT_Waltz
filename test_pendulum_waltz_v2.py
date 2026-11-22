"""Check Pendulum torque symmetry and count the offspring in a short run."""
import math
import random
import unittest
from unittest.mock import patch

try:
    import gymnasium
except ModuleNotFoundError:
    gymnasium = None


@unittest.skipUnless(gymnasium, "Install requirements-control.txt to run Pendulum checks")
class PendulumChecks(unittest.TestCase):
    def test_symmetry_and_zero_torque(self):
        from pendulum_waltz_v2 import action, features
        class Controller:
            def activate(self, x):
                return [1 / (1 + math.exp(-1 - sum(x)))]
        network = Controller()
        self.assertEqual(action(network, (1., 0., 0.), True), 0.)
        for angle, speed in ((.1, .7), (2., -2.4)):
            c, s = math.cos(angle), math.sin(angle)
            reflected = (c, -s, -speed)
            self.assertEqual(features(reflected), tuple(-v for v in features((c, s, speed))))
            self.assertEqual(action(network, reflected, True), -action(network, (c, s, speed), True))

    def test_refined_population_and_candidate_counts(self):
        import pendulum_waltz_v2 as model
        cfg = model.original.pilot.create_config(4, mutation_power=.1)
        fitness_calls = []
        def fitness(g, config):
            score = sum(n.bias for n in g.nodes.values())
            fitness_calls.append(score)
            return score
        state = random.getstate()
        with patch.object(model, "local_child", wraps=model.local_child) as local, \
             patch.object(model, "_insert_waltzer_candidate", wraps=model._insert_waltzer_candidate) as cross:
            best = model.run_refined(cfg, fitness, 5, 456, (1,))
        self.assertEqual(len(fitness_calls), 600)
        self.assertEqual(best.fitness, max(fitness_calls))
        # Four breeding rounds on four islands, with one crossover and three copies each.
        self.assertEqual(cross.call_count, 16)
        self.assertEqual(local.call_count, 48)
        self.assertEqual(state, random.getstate())
        self.assertIsNone(cfg.genome_config.node_indexer)


if __name__ == "__main__":
    unittest.main()
