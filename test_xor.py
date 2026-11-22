"""Run the XOR classification experiment with python test_xor.py."""

import argparse
from copy import deepcopy
from importlib.util import find_spec
from math import isfinite
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

import neat

from neat_genome import HomologousGenome, INITIALIZATION
from neat_waltz import run_waltz


XOR = (((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0))
ISLAND_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")

# Shared by all three classification scripts. NEAT-Python 2.0 requires
# explicit configuration and loads it from a file.
NEAT_SETTINGS = """
[NEAT]
pop_size = 30
fitness_criterion = max
fitness_threshold = 0.0
no_fitness_termination = True
reset_on_extinction = True

[HomologousGenome]
num_inputs = 2
num_outputs = 1
num_hidden = 0
feed_forward = True
initial_connection = full_direct
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.5
conn_add_prob = 0.4
conn_delete_prob = 0.1
node_add_prob = 0.2
node_delete_prob = 0.05
single_structural_mutation = False
structural_mutation_surer = default
activation_default = sigmoid
activation_options = sigmoid
activation_mutate_rate = 0.0
aggregation_default = sum
aggregation_options = sum
aggregation_mutate_rate = 0.0
enabled_default = True
enabled_mutate_rate = 0.01
enabled_rate_to_true_add = 0.0
enabled_rate_to_false_add = 0.0
bias_init_mean = 0.0
bias_init_stdev = 1.0
bias_init_type = gaussian
bias_mutate_rate = 0.8
bias_mutate_power = 0.5
bias_replace_rate = 0.1
bias_min_value = -30.0
bias_max_value = 30.0
weight_init_mean = 0.0
weight_init_stdev = 1.0
weight_init_type = gaussian
weight_mutate_rate = 0.8
weight_mutate_power = 0.5
weight_replace_rate = 0.1
weight_min_value = -30.0
weight_max_value = 30.0
response_init_mean = 1.0
response_init_stdev = 0.0
response_init_type = gaussian
response_mutate_rate = 0.0
response_mutate_power = 0.0
response_replace_rate = 0.0
response_min_value = 1.0
response_max_value = 1.0
time_constant_init_mean = 1.0
time_constant_init_stdev = 0.0
time_constant_init_type = gaussian
time_constant_mutate_rate = 0.0
time_constant_mutate_power = 0.0
time_constant_replace_rate = 0.0
time_constant_min_value = 1.0
time_constant_max_value = 1.0

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation = 15
species_elitism = 1

[DefaultReproduction]
elitism = 1
survival_threshold = 0.3
min_species_size = 1
"""


def run_classification(name, train, test=None, *, generations, make_test=None, make_train=None):
    """Train the Waltz model and report accuracy for both returned genomes."""
    parser = argparse.ArgumentParser(description=f"Test NEAT Waltz on {name} classification.")
    parser.add_argument("--generations", type=int, default=generations)
    parser.add_argument("--seed", type=int,
                        help="Evolution seed (default: 0; --compare: 0–4; other comparisons: 5–14)")
    parser.add_argument("--seeds", type=int, nargs=2, metavar=("FIRST", "LAST"),
                        help="Inclusive evolution seed range for comparisons")
    parser.add_argument("--test-seed", type=int,
                        help="Circle/Spiral test data seed (default: 11; --compare-baselines: 12)")
    parser.add_argument("--train-size", type=int,
                        help="Balanced Circle/Spiral training points, even and positive (default: 128)")
    parser.add_argument("--mutation-power", type=float, default=0.5,
                        help="Gaussian mutation standard deviation for weights and biases (default: 0.5)")
    tempo_options = parser.add_mutually_exclusive_group()
    tempo_options.add_argument("--tempo", type=int, help="Generations per island visit (default: 5; --compare: 1, 5, 15)")
    tempo_options.add_argument("--tempo-schedule", type=int, nargs="+", metavar="TEMPO",
                               help="Single run: tempos for successive complete tours, keeping the last (e.g. 15 10 5)")
    parser.add_argument("--waltzers", type=int, default=1, help="Number of simultaneous waltzers (1 through 4)")
    parser.add_argument("--candidates", type=int, default=1, help="Candidates per waltzer per generation (default: 1)")
    parser.add_argument("--hidden", type=int, default=0,
                        help="Initial hidden nodes per genome, shared by all compared methods (default: 0)")
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument("--compare", action="store_true", help="Compare crossover and mutation-only runs by tempo")
    comparison.add_argument("--compare-neat", action="store_true", help="Compare Waltz with standard single-population NEAT")
    comparison.add_argument("--compare-travel", action="store_true", help="Compare a traveling waltzer with one fixed on island A")
    comparison.add_argument("--compare-waltzers", action="store_true", help="Compare 1x1, 1x4 and 4x1 waltzers/candidates")
    comparison.add_argument("--compare-baselines", action="store_true",
                            help="Compare best genomes from classic NEAT, independent islands and Waltz (seeds 15–34)")
    comparison.add_argument("--compare-schedule", action="store_true",
                            help="Compare best genomes with fixed tempos 5, 15 and accelerating 15->10->5 (seeds 5–14)")
    parser.add_argument("--mutation-only", action="store_true", help="Mutate a clone of the waltzer for each candidate")
    parser.add_argument("--trace", action="store_true", help="Show the waltzer's progress per island visit")
    parser.add_argument("--plot", nargs="?", type=Path, const=Path(f"{name.lower()}_waltz.png"),
                        metavar="PATH", help="Save a graph of waltzer progress by island visit")
    args = parser.parse_args()
    comparing = (args.compare or args.compare_neat or args.compare_travel or args.compare_waltzers
                 or args.compare_baselines or args.compare_schedule)
    if args.generations < 1 or (args.tempo is not None and args.tempo < 1):
        parser.error("--generations and --tempo must be positive")
    if comparing and (args.mutation_only or args.trace or args.plot):
        parser.error("Use --mutation-only, --trace and --plot with single runs")
    if args.tempo_schedule is not None and (comparing or any(value < 1 for value in args.tempo_schedule)):
        parser.error("Use --tempo-schedule with positive tempos and a single run")
    if args.compare_schedule and args.tempo is not None:
        parser.error("--compare-schedule supplies its own fixed tempos and schedule")
    if args.seeds is not None and (not comparing or args.seed is not None or args.seeds[0] > args.seeds[1]):
        parser.error("Use --seeds FIRST LAST with a comparison, FIRST <= LAST, and without --seed")
    if args.test_seed is not None and make_test is None:
        parser.error("--test-seed requires a Circle or Spiral dataset generator")
    if args.train_size is not None:
        if make_train is None or args.train_size < 2 or args.train_size % 2:
            parser.error("--train-size requires Circle/Spiral and a positive even number of points")
        train = make_train(args.train_size // 2, seed=10)
    if not 1 <= args.waltzers <= len(ISLAND_COLORS):
        parser.error("--waltzers must be between 1 and 4")
    if not 1 <= args.candidates < 30:
        parser.error("--candidates must be between 1 and 29")
    if args.hidden < 0:
        parser.error("--hidden must be nonnegative")
    if not isfinite(args.mutation_power) or args.mutation_power < 0:
        parser.error("--mutation-power must be finite and nonnegative")
    if args.compare_waltzers and (args.waltzers != 1 or args.candidates != 1):
        parser.error("--compare-waltzers supplies its own waltzer and candidate counts")
    if args.plot and find_spec("matplotlib") is None:
        parser.error("--plot requires Matplotlib. Install it with python -m pip install matplotlib")

    if make_test is not None:
        test_seed = args.test_seed if args.test_seed is not None else (12 if args.compare_baselines else 11)
        test = make_test(test_seed)
        if set(inputs for inputs, _ in train) & set(inputs for inputs, _ in test):
            parser.error("Training and test inputs overlap; choose a different --test-seed")
        print(f"Test data seed: {test_seed} | {len(test)} test points", flush=True)
    if make_train is not None:
        print(f"Training data seed: 10 | {len(train)} training points", flush=True)

    with TemporaryDirectory() as directory:
        path = Path(directory) / "neat.ini"
        path.write_text(NEAT_SETTINGS, encoding="utf-8")
        config = neat.Config(HomologousGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation, path)
    config.genome_config.num_hidden = args.hidden
    config.genome_config.weight_mutate_power = args.mutation_power
    config.genome_config.bias_mutate_power = args.mutation_power
    print(f"Initial hidden nodes per genome: {args.hidden} | "
          f"Initial connections: {config.genome_config.initial_connection}", flush=True)
    print(f"Initialization: {INITIALIZATION}", flush=True)
    print(f"Gaussian mutation standard deviation (weights/biases): {args.mutation_power:g}", flush=True)

    def fitness(genome, config):
        network = neat.nn.FeedForwardNetwork.create(genome, config)
        return -sum((network.activate(inputs)[0] - target) ** 2 for inputs, target in train) / len(train)

    if comparing:
        if args.seeds is not None:
            seeds = range(args.seeds[0], args.seeds[1] + 1)
        elif args.seed is not None:
            seeds = (args.seed,)
        elif args.compare_baselines:
            seeds = range(15, 35)
        else:
            seeds = range(5) if args.compare else range(5, 15)
        examples = train if test is None else test
        split = "truth table" if test is None else "test set"
        print(f"{name} | accuracy on {split} | {args.generations} generations per run", flush=True)
        if args.compare:
            tempos = (1, 5, 15) if args.tempo is None else (args.tempo,)
            compare_tempos(config, fitness, args.generations, examples, tempos, seeds,
                           waltzers=args.waltzers, candidates_per_waltzer=args.candidates)
        else:
            tempo = 5 if args.tempo is None else args.tempo
            if args.compare_schedule:
                compare_tempo_schedule(config, fitness, args.generations, examples, seeds,
                                       waltzers=args.waltzers, candidates_per_waltzer=args.candidates)
            elif args.compare_baselines:
                compare_baselines(config, fitness, args.generations, examples, tempo, seeds,
                                  waltzers=args.waltzers, candidates_per_waltzer=args.candidates)
            elif args.compare_waltzers:
                compare_waltzer_counts(config, fitness, args.generations, examples, tempo, seeds)
            else:
                compare = compare_travel if args.compare_travel else compare_classic_neat
                compare(config, fitness, args.generations, examples, tempo, seeds,
                        waltzers=args.waltzers, candidates_per_waltzer=args.candidates)
        return

    seed = 0 if args.seed is None else args.seed
    tempo = 5 if args.tempo is None else args.tempo
    tempo_label = tempo if args.tempo_schedule is None else " → ".join(map(str, args.tempo_schedule))
    mode = "Mutation only" if args.mutation_only else "Crossover + mutation"
    visits = []

    def record_visit(**visit):
        if args.trace:
            print_visit(**visit)
        if args.plot:
            visits.append(visit)

    print(f"{name} | {args.generations} generations | seed {seed} | tempo {tempo_label} | {mode} | "
          f"{args.waltzers} waltzer(s), {args.candidates} candidate(s) each", flush=True)
    if args.trace and args.generations > 1:
        print("Waltzer / island | Generations | Training MSE in -> out | MSE drop | Accepted / tried"
              + (" | Tempo" if args.tempo_schedule is not None else ""), flush=True)
    best, waltzer = run_waltz(config, fitness, args.generations, islands=len(ISLAND_COLORS),
                             tempo=tempo, seed=seed, crossover=not args.mutation_only,
                             waltzers=args.waltzers, candidates_per_waltzer=args.candidates,
                             tempo_schedule=args.tempo_schedule,
                             on_visit=record_visit if args.trace or args.plot else None)
    waltzer_label = "Final waltzer" if args.waltzers == 1 else "Best final waltzer (training fitness)"
    for label, genome in (("Best genome", best), (waltzer_label, waltzer)):
        network = neat.nn.FeedForwardNetwork.create(genome, config)
        datasets = (("truth table", train),) if test is None else (("train", train), ("test", test))
        for split, examples in datasets:
            correct = count_correct(network, examples)
            print(f"{label} | {split}: {correct}/{len(examples)} ({correct / len(examples):.1%})")
    if args.plot:
        save_progress_plot(args.plot, f"{name} · {mode} · {args.waltzers}x{args.candidates}",
                           visits, waltzer.fitness, seed, tempo_label, initial_hidden=args.hidden)
        print(f"Progress graph saved to {args.plot}")


def count_correct(network, examples):
    return sum((network.activate(inputs)[0] >= 0.5) == target for inputs, target in examples)


def accuracy(genome, config, examples):
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    return count_correct(network, examples) / len(examples)


def run_classic_neat(config, fitness, generations, seed, *, on_generation=None):
    """Run one NEAT population with the same total population size as the islands."""
    classic_config = deepcopy(config)
    classic_config.pop_size *= len(ISLAND_COLORS)
    classic_config.no_fitness_termination = True
    generation, evaluations, best_fitness = 0, 0, float("-inf")

    def evaluate(genomes, config):
        nonlocal generation, evaluations, best_fitness
        for _, genome in genomes:
            genome.fitness = fitness(genome, config)
        evaluations += len(genomes)
        current = max(g.fitness for _, g in genomes)
        best_fitness = max(best_fitness, current)
        if on_generation is not None:
            on_generation(generation=generation, genome_evaluations=evaluations,
                          candidate_evaluations=0, population_sizes=[len(genomes)],
                          island_best_fitness=[current], best_fitness=best_fitness, lineages=[])
        generation += 1

    population = neat.Population(classic_config, seed=seed)
    return population.run(evaluate, generations), population.config


def compare_tempo_schedule(config, fitness, generations, examples, seeds, *, waltzers=1, candidates_per_waltzer=1):
    """Compare fixed tempos with one shared accelerating schedule at equal budgets."""
    island_count = len(ISLAND_COLORS)
    budget = generations * island_count * config.pop_size
    print(f"{budget:,} genome evaluations per run. {island_count} islands of {config.pop_size}, "
          f"{waltzers} waltzer(s), {candidates_per_waltzer} candidate(s) each. "
          "Tempos: fixed 5, fixed 15, accelerating 15->10->5.", flush=True)
    print("Primary: best genome seen, selected by training fitness. "
          "Schedule changes after complete tours; the last tempo continues. "
          "Generation 0 initializes waltzers; candidate evaluations start at 1.", flush=True)
    print("Seed   | Fixed 5 | Fixed 15 | 15->10->5 | vs 5 (pp) | vs 15 (pp)", flush=True)

    def print_row(seed, scores):
        fast, slow, accelerating, gain_fast, gain_slow = scores
        print(f"{seed!s:6} | {fast:7.2%} | {slow:8.2%} | {accelerating:8.2%} | "
              f"{gain_fast:+9.2f} | {gain_slow:+10.2f}", flush=True)

    rows = []
    for seed in seeds:
        scores = []
        for options in ({"tempo": 5}, {"tempo": 15}, {"tempo_schedule": (15, 10, 5)}):
            best, _ = run_waltz(config, fitness, generations, islands=island_count, seed=seed,
                                waltzers=waltzers, candidates_per_waltzer=candidates_per_waltzer, **options)
            scores.append(accuracy(best, config, examples))
        fast, slow, accelerating = scores
        row = (*scores, 100 * (accelerating - fast), 100 * (accelerating - slow))
        rows.append(row)
        print_row(seed, row)
    print_row("median", tuple(median(column) for column in zip(*rows)))
    for label, column in (("fixed 5", 3), ("fixed 15", 4)):
        wins = sum(row[column] > 0 for row in rows)
        ties = sum(row[column] == 0 for row in rows)
        print(f"Accelerating vs {label}: {wins} wins, {ties} ties, {len(rows) - wins - ties} losses", flush=True)


def compare_baselines(config, fitness, generations, examples, tempo, seeds, *, waltzers=1, candidates_per_waltzer=1):
    """Use best training fitness for all primary selections, then score on test data."""
    island_count = len(ISLAND_COLORS)
    budget = generations * island_count * config.pop_size
    print(f"{budget:,} genome evaluations per run. Classic: one population of {island_count * config.pop_size}; "
          f"independent/Waltz: {island_count} islands of {config.pop_size}. "
          f"Tempo {tempo}, {waltzers} waltzer(s), {candidates_per_waltzer} candidate(s) each.", flush=True)
    print("Primary: best genome seen, selected by training fitness. Final waltzer is secondary. "
          "Gains = Best Waltz minus baseline; medians use paired differences.", flush=True)
    print("Seed   | Classic NEAT | Independent | Best Waltz | vs NEAT (pp) | vs islands (pp) | Final waltzer", flush=True)

    def print_row(seed, scores):
        classic, independent, best, neat_gain, islands_gain, waltzer = scores
        print(f"{seed!s:6} | {classic:12.2%} | {independent:11.2%} | {best:10.2%} | "
              f"{neat_gain:+12.2f} | {islands_gain:+15.2f} | {waltzer:13.2%}", flush=True)

    rows = []
    for seed in seeds:
        classic_genome, classic_config = run_classic_neat(config, fitness, generations, seed)
        independent_genome, _ = run_waltz(config, fitness, generations, islands=island_count,
                                         seed=seed, waltzers=0)
        best_genome, final_waltzer = run_waltz(config, fitness, generations, islands=island_count,
                                              tempo=tempo, seed=seed, waltzers=waltzers,
                                              candidates_per_waltzer=candidates_per_waltzer)
        classic = accuracy(classic_genome, classic_config, examples)
        independent = accuracy(independent_genome, config, examples)
        best = accuracy(best_genome, config, examples)
        row = (classic, independent, best, 100 * (best - classic), 100 * (best - independent),
               accuracy(final_waltzer, config, examples))
        rows.append(row)
        print_row(seed, row)
    print_row("median", tuple(median(column) for column in zip(*rows)))
    for label, column in (("NEAT", 3), ("independent islands", 4)):
        wins = sum(row[column] > 0 for row in rows)
        ties = sum(row[column] == 0 for row in rows)
        print(f"Best Waltz vs {label}: {wins} wins, {ties} ties, {len(rows) - wins - ties} losses", flush=True)


def compare_waltzer_counts(config, fitness, generations, examples, tempo, seeds):
    """Separate the effects of producing more candidates and using more travelers."""
    island_count = len(ISLAND_COLORS)
    budget = generations * island_count * config.pop_size
    print(f"{budget:,} genome evaluations per run. Fixed tempo {tempo}. "
          "NxK = N waltzers, K candidates each. Waltzers are selected by training fitness.", flush=True)
    print("Seed   | 1x1 waltzer | 1x4 waltzer | 4x1 waltzer | Candidates gain (pp) | Travelers gain (pp)", flush=True)

    def print_row(seed, scores):
        one, more_candidates, more_waltzers, candidate_gain, traveler_gain = scores
        print(f"{seed!s:6} | {one:11.2%} | {more_candidates:11.2%} | {more_waltzers:11.2%} | "
              f"{candidate_gain:+20.2f} | {traveler_gain:+19.2f}", flush=True)

    rows = []
    for seed in seeds:
        scores = []
        for waltzers, candidates in ((1, 1), (1, 4), (4, 1)):
            _, selected_waltzer = run_waltz(config, fitness, generations, islands=island_count,
                                            tempo=tempo, seed=seed, waltzers=waltzers,
                                            candidates_per_waltzer=candidates)
            scores.append(accuracy(selected_waltzer, config, examples))
        one, more_candidates, more_waltzers = scores
        row = (*scores, 100 * (more_candidates - one), 100 * (more_waltzers - more_candidates))
        rows.append(row)
        print_row(seed, row)
    print_row("median", tuple(median(column) for column in zip(*rows)))


def compare_travel(config, fitness, generations, examples, tempo, seeds, *, waltzers=1, candidates_per_waltzer=1):
    """Compare traveling and fixed waltzers at the same evaluation budget."""
    island_count = len(ISLAND_COLORS)
    budget = generations * island_count * config.pop_size
    print(f"{budget:,} genome evaluations per run. {island_count} islands of {config.pop_size}, "
          f"{waltzers} waltzer(s), {candidates_per_waltzer} candidate(s) each, "
          f"crossover in both modes, travel tempo {tempo}. Fixed waltzers stay on their starting islands. "
          "Delta = traveling minus fixed.", flush=True)
    print("Seed   | Waltzer travel | Waltzer fixed   | Delta (pp) | Best travel | Best fixed   | Best delta (pp)",
          flush=True)

    def print_row(seed, scores):
        traveling, fixed, delta, best_traveling, best_fixed, best_delta = scores
        print(f"{seed!s:6} | {traveling:14.2%} | {fixed:15.2%} | {delta:+10.2f} | "
              f"{best_traveling:11.2%} | {best_fixed:12.2%} | {best_delta:+15.2f}", flush=True)

    rows = []
    for seed in seeds:
        scores = []
        for travel in (True, False):
            best, waltzer = run_waltz(config, fitness, generations, islands=island_count,
                                     tempo=tempo, seed=seed, travel=travel,
                                     waltzers=waltzers, candidates_per_waltzer=candidates_per_waltzer)
            scores.append((accuracy(waltzer, config, examples), accuracy(best, config, examples)))
        (traveling, best_traveling), (fixed, best_fixed) = scores
        row = (traveling, fixed, 100 * (traveling - fixed),
               best_traveling, best_fixed, 100 * (best_traveling - best_fixed))
        rows.append(row)
        print_row(seed, row)
    print_row("median", tuple(median(column) for column in zip(*rows)))


def compare_classic_neat(config, fitness, generations, examples, tempo, seeds, *, waltzers=1, candidates_per_waltzer=1):
    """Compare paired runs with the same total number of genomes per generation."""
    island_count = len(ISLAND_COLORS)
    def print_row(seed, scores):
        classic, best, delta_best, waltzer, delta_waltzer = scores
        print(f"{seed!s:6} | {classic:12.2%} | {best:10.2%} | {delta_best:+14.2f} | "
              f"{waltzer:13.2%} | {delta_waltzer:+17.2f}", flush=True)

    total_population = island_count * config.pop_size
    budget = generations * total_population
    print(f"{budget:,} genome evaluations per run. NEAT: {total_population} genomes in one population. "
          f"Waltz: {island_count} islands of {config.pop_size}, tempo {tempo}, "
          f"{waltzers} waltzer(s), {candidates_per_waltzer} candidate(s) each.", flush=True)
    print("Seed   | Classic NEAT | Best Waltz | Best diff (pp) | Final waltzer | Waltzer diff (pp)", flush=True)

    rows = []
    for seed in seeds:
        classic_genome, classic_config = run_classic_neat(config, fitness, generations, seed)
        classic = accuracy(classic_genome, classic_config, examples)
        best_genome, final_waltzer = run_waltz(config, fitness, generations, islands=island_count,
                                              tempo=tempo, seed=seed, waltzers=waltzers,
                                              candidates_per_waltzer=candidates_per_waltzer)
        best = accuracy(best_genome, config, examples)
        waltzer = accuracy(final_waltzer, config, examples)
        row = classic, best, 100 * (best - classic), waltzer, 100 * (waltzer - classic)
        rows.append(row)
        print_row(seed, row)
    print_row("median", tuple(median(column) for column in zip(*rows)))


def compare_tempos(config, fitness, generations, examples, tempos, seeds, *, waltzers=1, candidates_per_waltzer=1):
    """Print paired results and medians for each tempo."""
    island_count = len(ISLAND_COLORS)
    budget = generations * island_count * config.pop_size
    print(f"{budget:,} genome evaluations per run. {waltzers} waltzer(s), "
          f"{candidates_per_waltzer} candidate(s) each. Delta = crossover minus mutation only.", flush=True)
    print("Tempo | Seed   | Waltzer cross | Waltzer mut. | Delta (pp) | Best cross | Best mut.", flush=True)

    def print_row(tempo, seed, scores):
        cross, mutation, delta, best_cross, best_mutation = scores
        print(f"{tempo:5} | {seed!s:6} | {cross:13.2%} | {mutation:12.2%} | {delta:+10.2f} | "
              f"{best_cross:10.2%} | {best_mutation:9.2%}", flush=True)

    for tempo in tempos:
        rows = []
        for seed in seeds:
            scores = []
            for crossover in (True, False):
                best, waltzer = run_waltz(config, fitness, generations, islands=island_count,
                                         tempo=tempo, seed=seed, crossover=crossover,
                                         waltzers=waltzers, candidates_per_waltzer=candidates_per_waltzer)
                scores.append((accuracy(waltzer, config, examples), accuracy(best, config, examples)))
            (cross, best_cross), (mutation, best_mutation) = scores
            row = cross, mutation, 100 * (cross - mutation), best_cross, best_mutation
            rows.append(row)
            print_row(tempo, seed, row)
        print_row(tempo, "median", tuple(median(column) for column in zip(*rows)))


def print_visit(island, first_generation, last_generation, before, after, accepted,
                waltzer_id=None, candidates_per_generation=1, tempo=None):
    island_name = chr(ord("A") + island)
    location = island_name if waltzer_id is None else f"W{waltzer_id + 1} @ {island_name}"
    attempts = (last_generation - first_generation + 1) * candidates_per_generation
    print(f"{location:16} | {first_generation:3d}-{last_generation:<3d}     | "
          f"{-before:9.6f} -> {-after:9.6f}  | {after - before:.6g} | {accepted}/{attempts}"
          + (f" | {tempo}" if tempo is not None else ""), flush=True)


def save_progress_plot(path, name, visits, final_fitness, seed, tempo, *, initial_hidden=None):
    """Plot reductions in training MSE as percentages of the starting error."""
    from matplotlib.figure import Figure

    groups = {}
    for visit in visits:
        groups.setdefault(visit.get("waltzer_id", 0), []).append(visit)
    if not groups:
        groups[0] = []
    gains = [0.0] * len(ISLAND_COLORS)

    figure = Figure(figsize=(8.5, 6.4), layout="constrained")
    progress, islands = figure.subplots(2, 1, height_ratios=(2, 1))
    title = f"{name} | Waltzer progress"
    if initial_hidden is not None:
        title += f"\nInitial hidden nodes: {initial_hidden}"
    figure.suptitle(title, fontsize=17, fontweight="bold")

    maximum_reduction = 0
    for waltzer_id, group in groups.items():
        start = group[0]["before"] if group else final_fitness
        scale = 100 / -start if start < 0 else 0
        generations = [0] + [visit["last_generation"] for visit in group]
        reductions = [0] + [(visit["after"] - start) * scale for visit in group]
        maximum_reduction = max(maximum_reduction, max(reductions))
        line_color = ISLAND_COLORS[waltzer_id] if len(groups) > 1 else "#424B5A"
        progress.step(generations, reductions, where="post", color=line_color, linewidth=1.8,
                      label=f"Waltzer {waltzer_id + 1}")
        progress.scatter(generations[0], reductions[0], color=line_color, zorder=3)
        if group:
            progress.scatter(generations[1:], reductions[1:],
                             c=[ISLAND_COLORS[visit["island"]] for visit in group],
                             s=31, edgecolors="white", linewidths=0.5, zorder=3)
        for visit in group:
            gains[visit["island"]] += (visit["after"] - visit["before"]) * scale / len(groups)

    if len(groups) == 1:
        progress.set_title(f"Training MSE: {-start:.4g} to {-final_fitness:.4g} "
                           f"({reductions[-1]:.1f}% lower)", loc="left", fontsize=12)
    else:
        progress.set_title("Each line follows one independent waltzer", loc="left", fontsize=12)
        progress.legend(loc="upper left", fontsize=9)
    first_group = next(iter(groups.values()))
    for previous, current in zip(first_group, first_group[1:]):
        if current.get("tempo") != previous.get("tempo"):
            boundary = current["first_generation"] - 1
            progress.axvline(boundary, color="#666666", linestyle="--", linewidth=0.8)
            progress.text(boundary, 0.98, f"Tempo {current['tempo']} ", rotation=90,
                          transform=progress.get_xaxis_transform(), va="top", ha="right", fontsize=8)
    progress.set_xlabel(f"Generation · dots show host island colors · seed {seed}, tempo {tempo}")
    progress.set_ylabel("Initial error removed (%)")
    progress.set_ylim(0, max(5, maximum_reduction * 1.12))

    bars = islands.bar([f"Island {chr(ord('A') + i)}" for i in range(len(gains))], gains,
                       color=ISLAND_COLORS, width=0.56)
    islands.bar_label(bars, fmt="%.1f%%", padding=3)
    islands.set_title("Error reduction by island, averaged across waltzers", loc="left", fontsize=12)
    islands.set_ylabel("Mean % of initial error")
    islands.set_ylim(0, max(5, max(gains) * 1.25))

    for axes in (progress, islands):
        axes.spines[["top", "right"]].set_visible(False)
        axes.set_axisbelow(True)
        axes.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=150)


if __name__ == "__main__":
    run_classification("XOR", XOR, generations=120)
