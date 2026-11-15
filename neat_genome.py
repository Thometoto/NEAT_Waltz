"""Keep corresponding initial hidden nodes homologous in neat-python 2.0."""

from copy import copy
from itertools import count

import neat


INITIALIZATION = "shared_initial_hidden_ids_v1"


class HomologousGenome(neat.DefaultGenome):
    """Use shared initial node IDs, then the usual global mutation allocator."""

    def configure_new(self, config):
        if config.num_hidden == 0:
            super().configure_new(config)
            return

        # Reuse the same hidden IDs in every fresh genome, including on other
        # islands. Weights and biases still receive independent random draws.
        initial_config = copy(config)
        initial_config.node_indexer = count(max(config.output_keys) + 1)
        super().configure_new(initial_config)

        # Reserve those IDs for the initial topology. Do not rewind the global
        # allocator when creating another genome or repopulating an island.
        if config.node_indexer is None:
            config.node_indexer = count(max(self.nodes) + 1)
