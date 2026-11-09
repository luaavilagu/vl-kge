"""DEAP search for global visual, textual, and structural contributions.

An individual contains three nonnegative weights summing to one, ordered
visual, textual, structural. There are two independent degrees of freedom.
Fitness is a one-element tuple containing filtered validation MRR.

This module defines the search without loading datasets or training models.
Use ``python -m vlkge.ga.run_fusion_ga --help`` for the checkpoint workflow.
"""

from dataclasses import asdict, dataclass
import math
import random

from deap import algorithms, base, creator, tools
from vlkge.models.mas_gatranse import MODALITIES, EQUAL_WEIGHTS, validate_weights


# DEAP stores generated classes globally. Guards allow notebook/module reloads.
if not hasattr(creator, "FusionFitnessMax"):
    creator.create("FusionFitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "FusionIndividual"):
    creator.create("FusionIndividual", list, fitness=creator.FusionFitnessMax)


@dataclass(frozen=True)
class GAConfig:
    """Starting settings for a single-objective, elitist generational GA.

    Crossover is attempted per parent pair; mutation is attempted per child.
    They are independent stages, so a child may undergo both. Random reset
    replaces the entire weight vector when mutation is invoked. Set patience
    to None to disable early stopping. These are defaults, not tuned optima.
    """

    population_size: int = 30
    generations: int = 30
    tournament_size: int = 3
    crossover: str = "arithmetic"
    crossover_probability: float = 0.8
    mutation_probability: float = 0.2
    sbx_eta: float = 20.0
    elite_size: int = 2
    patience: int = 10
    seed: int = 42

    def __post_init__(self):
        """Reject settings that cannot form a valid evolving population."""
        for name in ("population_size", "generations", "tournament_size", "elite_size", "seed"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer.")
        if self.population_size < 4 or self.generations < 0:
            raise ValueError("Use at least 4 individuals and nonnegative generations.")
        if not 1 <= self.elite_size < self.population_size:
            raise ValueError("elite_size must be positive and below population_size.")
        if not 1 <= self.tournament_size <= self.population_size:
            raise ValueError("tournament_size must be within the population size.")
        if self.crossover not in ("arithmetic", "sbx"):
            raise ValueError("crossover must be 'arithmetic' or 'sbx'.")
        for probability in (self.crossover_probability, self.mutation_probability):
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Variation probabilities must be in [0, 1].")
        if not math.isfinite(self.sbx_eta) or self.sbx_eta <= 0:
            raise ValueError("sbx_eta must be finite and positive.")
        if self.patience is not None and (type(self.patience) is not int or self.patience < 1):
            raise ValueError("patience must be a positive integer or None.")


def reference_weights():
    """Seed equal thirds and all three single modalities, including boundaries."""
    return [EQUAL_WEIGHTS, (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]


def sample_weights(generator=random):
    """Sample uniformly over the weight triangle (Dirichlet(1,1,1)).

    Divide three independent unit-rate exponential draws by their sum.
    Python's generator keeps DEAP seeding reproducible. Normalizing three
    uniform draws instead would not sample this triangle uniformly.
    """
    values = [generator.expovariate(1.) for _ in MODALITIES]
    total = math.fsum(values)
    return tuple(value / total for value in values) if total else EQUAL_WEIGHTS


def named_weights(weights):
    """Return a validated dictionary in the explicit modality order."""
    return dict(zip(MODALITIES, validate_weights(weights)))


def arithmetic_crossover(first, second):
    """Mix complete parental vectors using one shared coefficient.

    Save both original genes before writing either child. Convex combinations
    keep both children between their parents and therefore within [0, 1].
    DEAP expects the modified pair to be returned.
    """
    a, b = tuple(first), tuple(second)
    coefficient = random.random()
    first[:] = validate_weights([coefficient * x + (1 - coefficient) * y for x, y in zip(a, b)])
    second[:] = validate_weights([(1 - coefficient) * x + coefficient * y for x, y in zip(a, b)])
    return first, second


def bounded_sbx_crossover(first, second, eta):
    """Bound SBX coordinates, then repair each child's sum-to-one constraint.

    Normalize positive totals, using equal thirds for an all-zero child.
    Per-coordinate bounds alone do not enforce the required vector sum.
    """
    tools.cxSimulatedBinaryBounded(first, second, eta=eta, low=0., up=1.)
    for child in (first, second):
        total = math.fsum(child)
        child[:] = [value / total for value in child] if total else EQUAL_WEIGHTS
    return first, second


def random_reset_mutation(individual):
    """Reset the whole vector uniformly on the simplex; return a one-item tuple.

    This supplies global exploration even when arithmetic crossover only
    explores between parents. The outer evolution loop controls how often
    mutation is invoked, so no second per-gene probability is necessary.
    """
    individual[:] = sample_weights()
    return (individual,)


def build_toolbox(evaluate_weights, config):
    """Register individual/population creation, evaluation, and operators.

    evaluate_weights accepts three normalized weights and returns validation MRR.
    This adapter validates it and wraps it in DEAP's required fitness tuple.
    Tournament selection supplies parents; selBest is reserved for elites.
    """
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.FusionIndividual, sample_weights)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=config.tournament_size)
    if config.crossover == "arithmetic":
        toolbox.register("mate", arithmetic_crossover)
    else:
        toolbox.register("mate", bounded_sbx_crossover, eta=config.sbx_eta)
    toolbox.register("mutate", random_reset_mutation)

    def evaluate(individual):
        """Validate the vector and convert scalar MRR to DEAP fitness."""
        mrr = float(evaluate_weights(validate_weights(individual)))
        if not math.isfinite(mrr) or not 0 <= mrr <= 1:
            raise ValueError("The evaluator must return finite MRR in [0, 1].")
        return (mrr,)

    toolbox.register("evaluate", evaluate)
    return toolbox


def run_ga(evaluate_weights, config=None, progress=None):
    """Evolve normalized contributions with elitism and cached fitness.

    The initial population includes equal thirds and three single modalities.
    Remaining individuals are uniform simplex draws. Each generation copies the best elites
    unchanged and fills other positions using tournament-selected parents.
    DEAP varAnd clones parents and invalidates fitness after variation.

    HallOfFame records the best-ever individual; copying elites separately
    guarantees survival in the actual population. Repeated weight vectors reuse
    cached fitness. The result distinguishes fitness requests from unique
    evaluations so random/grid-search budgets can be compared honestly.

    Seeding occurs before initialization. DEAP uses Python's global random
    generator; its original state is restored before this function returns.
    Model weights must remain fixed for the whole run or cached MRR is stale.
    """
    config = config or GAConfig()
    original_random_state = random.getstate()
    random.seed(config.seed)
    memo = {}
    requests = 0

    def memoized_evaluate(weights):
        """Avoid calculating the same deterministic fitness more than once."""
        nonlocal requests
        requests += 1
        key = validate_weights(weights)
        if key not in memo:
            value = float(evaluate_weights(key))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("The evaluator must return finite MRR in [0, 1].")
            memo[key] = value
        return memo[key]

    try:
        toolbox = build_toolbox(memoized_evaluate, config)
        population = [creator.FusionIndividual(w) for w in reference_weights()]
        population.extend(toolbox.population(n=config.population_size - 4))
        hall = tools.HallOfFame(1)
        history = []

        def evaluate_invalid(individuals):
            """Evaluate only new/changed genomes; unchanged clones keep fitness."""
            for individual in individuals:
                if not individual.fitness.valid:
                    individual.fitness.values = toolbox.evaluate(individual)

        def record(generation):
            """Record fitness, vector diversity, and cumulative evaluation counts."""
            values = [individual.fitness.values[0] for individual in population]
            row = {
                "generation": generation,
                "best_mrr": hall[0].fitness.values[0],
                "mean_mrr": sum(values) / len(values),
                **{f"best_{name}_weight": value for name, value in named_weights(hall[0]).items()},
                "unique_weights": len({tuple(ind) for ind in population}),
                "fitness_requests": requests,
                "unique_evaluations": len(memo),
            }
            history.append(row)
            if progress is not None:
                progress(dict(row))

        evaluate_invalid(population)
        hall.update(population)
        record(0)
        stale = 0
        stopping_reason = "generation_limit"
        for generation in range(1, config.generations + 1):
            previous_best = hall[0].fitness.values[0]
            elites = [toolbox.clone(ind) for ind in tools.selBest(population, config.elite_size)]
            parents = toolbox.select(population, config.population_size - config.elite_size)
            offspring = algorithms.varAnd(
                parents, toolbox, config.crossover_probability, config.mutation_probability
            )
            evaluate_invalid(offspring)
            population = elites + offspring
            hall.update(population)
            record(generation)
            stale = 0 if hall[0].fitness.values[0] > previous_best else stale + 1
            if config.patience is not None and stale >= config.patience:
                stopping_reason = "no_improvement"
                break

        return {
            "best_weights": named_weights(hall[0]),
            "best_mrr": float(hall[0].fitness.values[0]),
            "config": asdict(config),
            "history": history,
            "generations_completed": history[-1]["generation"],
            "stopping_reason": stopping_reason,
            "fitness_requests": requests,
            "unique_evaluations": len(memo),
        }
    finally:
        random.setstate(original_random_state)



    










