"""DEAP search for one global visual/textual fusion weight.

An individual is [alpha], decoded as (alpha, 1-alpha). Its fitness is a
one-element tuple containing validation MRR. The positive DEAP fitness
weight means maximize MRR; it is not a modality-fusion weight.

This module defines the search without loading datasets or training models.
Use ``python -m vlkge.ga.run_fusion_ga --help`` for the checkpoint workflow.
"""

from dataclasses import asdict, dataclass
import math
import random

from deap import algorithms, base, creator, tools


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
    always replaces the single gene when mutation is invoked. Set patience
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
        if self.population_size < 3 or self.generations < 0:
            raise ValueError("Use at least 3 individuals and nonnegative generations.")
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


def arithmetic_crossover(first, second):
    """Mix parental alpha values in place using one uniform coefficient.

    Save both original genes before writing either child. Convex combinations
    keep both children between their parents and therefore within [0, 1].
    DEAP expects the modified pair to be returned.
    """
    a, b = first[0], second[0]
    coefficient = random.random()
    first[0] = coefficient * a + (1 - coefficient) * b
    second[0] = (1 - coefficient) * a + coefficient * b
    return first, second


def random_reset_mutation(individual):
    """Replace alpha with a fresh uniform value, returning a one-item tuple.

    This supplies global exploration even when arithmetic crossover only
    explores between parents. The outer evolution loop controls how often
    mutation is invoked, so no second per-gene probability is necessary.
    """
    individual[0] = random.uniform(0.0, 1.0)
    return (individual,)


def build_toolbox(evaluate_alpha, config):
    """Register individual/population creation, evaluation, and operators.

    evaluate_alpha accepts a Python alpha and returns scalar validation MRR.
    This adapter validates it and wraps it in DEAP's required fitness tuple.
    Tournament selection supplies parents; selBest is reserved for elites.
    """
    toolbox = base.Toolbox()
    toolbox.register("attr_flt", random.uniform, 0.0, 1.0)
    toolbox.register("individual", tools.initRepeat, creator.FusionIndividual, toolbox.attr_flt, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=config.tournament_size)
    if config.crossover == "arithmetic":
        toolbox.register("mate", arithmetic_crossover)
    else:
        toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                         eta=config.sbx_eta, low=0.0, up=1.0)
    toolbox.register("mutate", random_reset_mutation)

    def evaluate(individual):
        """Validate the single bounded gene and convert MRR to DEAP fitness."""
        if len(individual) != 1 or not 0 <= individual[0] <= 1:
            raise ValueError("An individual must contain one alpha in [0, 1].")
        mrr = float(evaluate_alpha(float(individual[0])))
        if not math.isfinite(mrr) or not 0 <= mrr <= 1:
            raise ValueError("The evaluator must return finite MRR in [0, 1].")
        return (mrr,)

    toolbox.register("evaluate", evaluate)
    return toolbox


def run_ga(evaluate_alpha, config=None, progress=None):
    """Evolve alpha with explicit elite preservation and cached fitness.

    The initial population includes alpha 0, 0.5, and 1. The remaining
    individuals are uniform draws. Each generation copies the best elites
    unchanged and fills other positions using tournament-selected parents.
    DEAP varAnd clones parents and invalidates fitness after variation.

    HallOfFame records the best-ever individual; copying elites separately
    guarantees survival in the actual population. Repeated alpha values reuse
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

    def memoized_evaluate(alpha):
        """Avoid calculating the same deterministic fitness more than once."""
        nonlocal requests
        requests += 1
        if alpha not in memo:
            value = float(evaluate_alpha(alpha))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("The evaluator must return finite MRR in [0, 1].")
            memo[alpha] = value
        return memo[alpha]

    try:
        toolbox = build_toolbox(memoized_evaluate, config)
        population = toolbox.population(n=config.population_size - 3)
        population.extend(creator.FusionIndividual([a]) for a in (0.0, 0.5, 1.0))
        hall = tools.HallOfFame(1)
        history = []

        def evaluate_invalid(individuals):
            """Evaluate only new/changed genomes; unchanged clones keep fitness."""
            for individual in individuals:
                if not individual.fitness.valid:
                    individual.fitness.values = toolbox.evaluate(individual)

        def record(generation):
            """Record fitness, alpha diversity, and cumulative evaluation counts."""
            values = [individual.fitness.values[0] for individual in population]
            row = {
                "generation": generation,
                "best_mrr": hall[0].fitness.values[0],
                "mean_mrr": sum(values) / len(values),
                "best_alpha": float(hall[0][0]),
                "unique_alphas": len({float(ind[0]) for ind in population}),
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
            "best_alpha": float(hall[0][0]),
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



    










