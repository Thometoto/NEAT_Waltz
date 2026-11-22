# Implementation and observation audit

Scope: the existing Waltz mechanism and the tempo experiment runner. No new
partner-selection, tempo or acceptance rule is introduced.

## Confirmed behavior

- Ordinary NEAT populations reproduce independently. Active waltzers are retained
  outside those populations. Their child replaces an ordinary offspring slot,
  preferring an unevaluated offspring over an elite. Island sizes remain fixed.
- A local champion is selected on training fitness. Crossover and mutation create
  a child; the next population evaluation scores it. In Sine, local champions are
  chosen by regional fitness, but both crossover parents and lineage acceptance
  use their comparable global training fitness.
- If several children belong to one waltzer in a generation, only the best child
  can replace it. Improvement must be strict. Equal-fitness children are rejected
  as waltzers, but remain ordinary population members until ordinary selection.
- With `G` evaluated generations, `I` islands, population size `P`, `W` waltzers
  and `K` children per waltzer, the budget is `G * I * P`. Exactly
  `(G - 1) * W * K` of these evaluations are Waltz candidates. Initial retained
  waltzers are copied from evaluated champions, without an extra evaluation.
- Generation 0 initializes the lineages. Candidate evaluation rounds start at 1.
  The final visit can be truncated by the budget. A one-generation run produces
  no candidate and no visit summary. Stationary lineages keep their original host.
- Initial corresponding hidden-node IDs are shared. Islands share an innovation
  counter and a node allocator, allowing crossover without island-local ID
  collisions. The existing NEAT-Python reproduction policy for innovation-cache
  resets is retained.

## Observations

`run_waltz` and `run_domains` accept optional `on_candidate(**event)` and
`on_generation(**event)` callbacks. Their existing return values and `on_visit`
callback arguments remain compatible. `run_classic_neat` also accepts the
generation callback. The observers read already evaluated training scores.
Callbacks must not modify evolution state or draw from the evolution RNG.

`confirm_tempo.py` persists these events as `candidate_history` and
`generation_history`, retaining the old `visits` and regional `history` fields.
Metadata labels this schema `waltz_training_observations_v1`.

Candidate events contain:

| Field | Meaning |
| --- | --- |
| `generation` / `created_generation` | Evaluation round / preceding reproduction round |
| `waltzer_id` | Stable role index within this run |
| `island` / `tempo` | Host of this encounter / intended visit duration |
| `waltzer_parent` / `local_parent` | Parent snapshots at child creation; the local parent is null in mutation-only mode |
| `candidate` | Evaluated child snapshot |
| `fitness_before` | Incumbent waltzer's comparable training fitness |
| `fitness_gain` | Child fitness minus incumbent fitness; this can be negative |
| `accepted` | Whether this child actually took the waltzer role |

Genome snapshots contain origin island, local key, training fitness and topology
counts. Sine snapshots also contain `region_mse`. Origin and current host are
distinct: a genome can visit another island without changing its origin.
No full genotype checkpoint is saved by these observations.

Generation events contain cumulative genome/candidate evaluations, actual
population sizes, best training fitness so far and every active lineage after
acceptance. `island_best_fitness` measures the island's own selection criterion.
Sine also reports `island_best_global_fitness` for comparisons across domains.
Scores from different criteria must not be compared directly.

## Defensive changes

- Reject noninteger generation, island and tempo counts in `run_waltz`.
- Detect changed population sizes and training/candidate budget mismatches.
- Deep-copy the best global genome when archiving it in `run_waltz`, keeping the
  archived state independent of the live population.

## Verification

Validation on 2026-11-18: all 10 unit checks passed. The complete software smoke
workflow passed all 18 runs (six methods on three tasks), with 121 generation
records and 14,520 training genome evaluations per run. Saved JSON checks verified
candidate counts, visit acceptance totals, archive scores and all three tempo phases.

The unit checks cover visit boundaries, a scheduled tempo transition, truncated
visits, initialization-only runs, multiple simultaneous waltzers and multiple
children, strict acceptance and ties, global versus regional criteria, fixed
population/evaluation budgets, initial homology and restoration of configuration
and RNG state. They also compare runs with and without observers.

An additional before/after check compared 12 short deterministic runs against
references captured before editing: returned genome genes, fitness, visit events
and regional histories matched exactly with observers enabled. These checks
establish software behavior for the tested cases, not an advantage of Waltz or
proof that useful capabilities transfer between populations.
