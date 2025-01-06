from copy import deepcopy

import numpy as np

from tqdm import tqdm
from gxgp import Node, DagGP, gxgp_random
from src.gxgp import arity


class TrainingParameters:
    # Training data
    x: np.ndarray
    y: np.ndarray

    # EvAlg parameters
    num_generations: int
    population_size: int

    # Crossover parameters
    weighted_parent_selection: bool
    elitist_population: bool

    # Chance parameters
    random_event_chance: float

    def __init__(self,
            x: np.ndarray,
            y: np.ndarray,

            num_generations: int,
            population_size: int,

            weighted_parent_selection: bool = False,
            elitist_population: bool = False,

            random_event_chance: float = 0.0,
    ):
        self.x = x
        self.y = y

        self.num_generations = num_generations
        self.population_size = population_size

        self.weighted_parent_selection = weighted_parent_selection
        self.elitist_population = elitist_population

        self.random_event_chance = random_event_chance

    def __str__(self):
        return (f"Num. generations: {self.num_generations} \n"
                f"Population size: {self.population_size} \n"
                f"Weighted parent selection: {self.weighted_parent_selection} \n"
                f"Elitist population: {self.elitist_population} \n"
                f"Random event chance: {self.random_event_chance}")


class Fitness:
    individual: Node
    parameters: TrainingParameters

    def __init__(self, node: Node, parameters: TrainingParameters):
        self.individual = node
        self.parameters = parameters

    def __lt__(self, other):
        self_mse: float = DagGP.mse(self.individual, self.parameters.x, self.parameters.y)
        other_mse: float = DagGP.mse(other.individual, other.parameters.x, other.parameters.y)

        if gxgp_random.random() < self.parameters.random_event_chance:
            return len(self.individual.subtree) < len(other.individual.subtree)

        return self_mse < other_mse


    @property
    def mse(self):
        return DagGP.mse(self.individual, self.parameters.x, self.parameters.y)


def is_valid(individual: Node, parameters: TrainingParameters) -> bool:
    try:
        return (
            any(sub_node.is_variable for sub_node in individual.subtree) and
            not np.any(np.isnan(DagGP.evaluate(individual, parameters.x))) and
            not np.any(np.isinf(DagGP.evaluate(individual, parameters.x)))
        ) and (
            parameters.random_event_chance > 0 or
            len(individual.subtree) <= 10
        )
    except BaseException:
        return False


def cleanup(individual: Node) -> Node:
    if any([(not node.is_leaf and not is_cleaned_up(node)) for node in individual.successors]):
        individual.successors = [cleanup(successor) for successor in individual.successors]

        return cleanup(individual)

    param1: Node | None = None
    param2: Node | None = None

    if len(individual.successors) > 0:
        param1 = individual.successors[0]
    if len(individual.successors) > 1:
        param2 = individual.successors[1]

    if individual.short_name == "add":
        if param1.long_name.strip("-").isnumeric() and param2.long_name.strip("-").isnumeric():
            return Node(param1.value + param2.value)

        if param1.is_constant and param1.value == 0:
            return param2
        if param2.is_constant and param2.value == 0:
            return param1

    if individual.short_name == "subtract":
        if param1.long_name.strip("-").isnumeric() and param2.long_name.strip("-").isnumeric():
            return Node(param1.value - param2.value)

        if param2.is_constant and param2.value == 0:
            return param1

        if param1.long_name == param2.long_name:
            return Node(0)

    if individual.short_name == "multiply":
        if param1.long_name.strip("-").isnumeric() and param2.long_name.strip("-").isnumeric():
            return Node(param1.value * param2.value)

        if (param1.is_constant and param1.value == 0) or (param2.is_constant and param2.value == 0):
            return Node(0)

        if param1.is_constant and param1.value == 1:
            return param2
        if param2.is_constant and param2.value == 1:
            return param1

    if individual.short_name == "divide":
        if param1.long_name.strip("-").isnumeric() and param2.long_name.strip("-").isnumeric() and param2.value != 0:
            return Node(param1.value / param2.value)

        if param1.is_constant and param1.value == 0 and not (param2.is_constant and param2.value == 0):
            return Node(0)

        if param2.is_constant and param2.value == 1:
            return param1

        if param1.long_name == param2.long_name:
            return Node(1)

    if individual.short_name == "power":
        if (
            (param1.long_name.strip("-").isnumeric() and param2.long_name.strip("-").isnumeric()) and not
            (param1.is_constant and param1.value == 0 and param2.is_constant and param2.value < 0)
        ):
            return Node(param1.value ** param2.value)

        if param1.is_constant and param1.value == 0:
            return Node(0)

        if (param1.is_constant and param1.value == 1) or (param2.is_constant and param2.value == 0):
            return Node(1)

        if param2.is_constant and param2.value == 1:
            return param1

    if individual.short_name == "sin":
        if param1.is_constant and param1.value == 0:
            return Node(0)

        if param1.is_constant and param1.value == np.pi:
            return Node(1)

    if individual.short_name == "cos":
        if param1.is_constant and param1.value == 0:
            return Node(1)

        if param1.is_constant and param1.value == np.pi:
            return Node(0)

    if individual.short_name == "tan" and (
        (param1.is_constant and param1.value == 0) or
        (param1.is_constant and param1.value == np.pi)
    ):
        return Node(0)

    if individual.short_name == "exp" and param1.is_constant and param1.value == 0:
        return Node(1)

    if individual.short_name == "log":
        if param1.is_constant and param1.value == 1:
            return Node(0)

        if param1.is_constant and param1.value == np.e:
            return Node(1)

    return individual

def is_cleaned_up(individual: Node) -> bool:
    for sub_node in individual.subtree:
        param1: Node | None = None
        param2: Node | None = None

        if len(sub_node.successors) > 0:
            param1 = sub_node.successors[0]
        if len(sub_node.successors) > 1:
            param2 = sub_node.successors[1]

        if sub_node.short_name == "add" and (
            (param1.long_name.strip("-").isnumeric() and param2.long_name.strip("-").isnumeric()) or
            (param1.is_constant and param1.value == 0) or
            (param2.is_constant and param2.value == 0)
        ):
            return False

        if sub_node.short_name == "subtract" and (
            (param1.long_name.strip("-").isnumeric() and param2.long_name.strip("-").isnumeric()) or
            (param2.is_constant and param2.value == 0) or
            (param1.long_name == param2.long_name)
        ):
            return False

        if sub_node.short_name in ["multiply", "power"] and (
            (param1.long_name.strip("-").isnumeric() and param2.long_name.strip("-").isnumeric()) or
            (param1.is_constant and param1.value == 0) or
            (param2.is_constant and param2.value == 0) or
            (param1.is_constant and param1.value == 1) or
            (param2.is_constant and param2.value == 1)
        ):
            return False

        if sub_node.short_name == "divide" and (
            (param1.long_name.strip("-").isnumeric() and param2.long_name.strip("-").isnumeric() and param2.value != 0) or
            (param1.is_constant and param1.value == 0 and not (param2.is_constant and param2.value == 0)) or
            (param2.is_constant and param2.value == 1) or
            (param1.long_name == param2.long_name)
        ):
            return False

        if any([sub_node.short_name == "cos", sub_node.short_name == "sin", sub_node.short_name == "tan"]) and (
            (param1.is_constant and param1.value == 0) or
            (param1.is_constant and param1.value == np.pi)
        ):
            return False

        if sub_node.short_name == "exp" and param1.is_constant and param1.value == 0:
            return False

        if sub_node.short_name == "log" and (
            (param1.is_constant and param1.value == 1) or
            (param1.is_constant and param1.value == np.e)
        ):
            return False

    return True


def fitness(individual: Node, parameters: TrainingParameters) -> Fitness:
    return Fitness(individual, parameters)

def select_parents(parent_pool: list[Node], dag: DagGP, parameters: TrainingParameters) -> (Node, Node):
    match len(parent_pool):
        case 0:
            return ()

    p: np.ndarray = gxgp_random.uniform(size=len(parent_pool))

    if parameters.weighted_parent_selection:
        p = np.array([fitness(parent, parameters).mse for parent in parent_pool])

    p = p / p.sum()

    return tuple(gxgp_random.choice(parent_pool, size=2, p=p, replace=False))

def crossover(parents: (Node, Node)) -> Node:
    parents_with_successors: list[Node] = list(filter(lambda parent: len(parent.successors) != 0, parents))

    if len(parents_with_successors) == 0:
        return gxgp_random.choice(parents)
    if len(parents_with_successors) == 1:
        return parents_with_successors[0]

    offspring: Node = deepcopy(parents[0])
    if len(offspring.successors) == 0:
        offspring = deepcopy(parents[1])

    successors: list[Node] | None = None
    pivot: Node | None = None

    while successors is None or len(successors) == 0:
        pivot = gxgp_random.choice(list(offspring.subtree))
        successors = pivot.successors

    new_subtree: Node | None = None
    while new_subtree is None or len(new_subtree.successors) == 0:
        new_subtree = gxgp_random.choice(list(parents[1].subtree))

    successors[gxgp_random.integers(len(successors))] = new_subtree
    pivot.successors = successors

    return offspring


def mutate(individual: Node, dag: DagGP, parameters: TrainingParameters) -> Node:
    mutated_individual: Node = deepcopy(individual)

    for _ in range(len(mutated_individual.subtree) // 2):
        pivot: Node | None = None
        successors: list[Node] | None = None

        while pivot is None or len(successors) == 0:
            pivot = gxgp_random.choice(list(mutated_individual.subtree))
            successors = pivot.successors

        index: int = gxgp_random.integers(len(successors))

        if successors[index].is_leaf or gxgp_random.random() < parameters.random_event_chance:
            successors[index] = gxgp_random.choice(dag.params_pool)
        else:
            same_arity_operators: list[np.ufunc] = list(filter(lambda op: arity(op) == successors[index].arity, dag.operators))
            chosen_operator: np.ufunc = gxgp_random.choice(same_arity_operators)

            successors[index] = Node(chosen_operator, successors[index].successors)

        pivot.successors = successors

    return mutated_individual


def check_population(population: list[Node], parameters: TrainingParameters) -> Node | None:
    for individual in population:
        if DagGP.mse(individual, parameters.x, parameters.y) == 0:
            return individual

    return None


def train_model(parameters: TrainingParameters, verbose: bool = False) -> Node:
    dag: DagGP = DagGP(
        [
            np.add, np.subtract,
            np.multiply, np.divide,
            np.pow,
            np.sin, np.cos, np.tan,
            np.exp, np.log
        ],
        parameters.x.shape[0],
        [-1, 0, 1, 2, 5, np.pi, np.e]
    )

    # Create an initial population of size N
    population: list[Node] = []
    for _ in range(parameters.population_size * 10):
        new_individual: Node = cleanup(dag.create_individual())

        if is_valid(new_individual, parameters):
            population.append(new_individual)

        if len(population) == parameters.population_size:
            break

    # If any of the initial solutions already has an MSE value of 0.0 (extremely unlikely, but possible), returns
    # that solution and skips the training completely.
    final_solution: Node | None = check_population(population, parameters)
    if final_solution is not None:
        return final_solution

    if verbose:
        for n in population:
            print(n)

        print(all([is_valid(i, parameters) for i in population]))
        print()

    for i in range(parameters.num_generations):
        # Sort the population by fitness
        sorted_population: list[Node] = sorted(population, key=lambda ind: fitness(ind, parameters))

        if verbose:
            print(f"Generation {i}: completed sorting")
            print()

        # Restrict the pool of available parents to the best N / 2 individuals
        parents_pool = sorted_population[:parameters.population_size // 2]

        if verbose:
            print(f"Generation {i}: generated parents pool")

            for par in parents_pool:
                print(par)

            print(all([is_valid(p, parameters) for p in parents_pool]))
            print()

        # Create a new generation
        population = []
        if parameters.elitist_population:
            population.extend(sorted_population[:parameters.elitist_population // 10])

            for _ in range(parameters.population_size * 10):
                parents: tuple[Node, Node] = select_parents(parents_pool, dag, parameters)
                offspring: Node = cleanup(crossover(parents))

                if is_valid(offspring, parameters):
                    population.append(offspring)

                if len(population) == parameters.population_size:
                    break
        else:
            # Fill the population of this generation with children generated by crossover
            for _ in range(parameters.population_size * 10):
                parents: tuple[Node, Node] = select_parents(parents_pool, dag, parameters)
                offspring: Node = cleanup(crossover(parents))

                if is_valid(offspring, parameters):
                    population.append(offspring)

                if len(population) == parameters.population_size:
                    break

        if verbose:
            print(f"Generation {i}: created new generation")

            for n in population:
                print(n)

            print(all([is_valid(i, parameters) for i in population]))
            print()

        # Check for the (extremely unlikely) case where a solution has MSE of 0.0.
        # If it does, stop the computations and return that solution.
        final_solution: Node | None = check_population(population, parameters)
        if final_solution is not None:
            return final_solution

        # Apply mutations to the individuals
        for index, individual in enumerate(population):
            for _ in range(10):
                mutated_individual: Node = cleanup(mutate(individual, dag, parameters))

                if is_valid(mutated_individual, parameters):
                    population[index] = mutated_individual

        if verbose:
            print(f"Generation {i}: mutated new generation")

            for n in population:
                print(n)

            print(all([is_valid(i, parameters) for i in population]))
            print()

        # Check again for the stopping case.
        final_solution: Node | None = check_population(population, parameters)
        if final_solution is not None:
            return final_solution

    return sorted(population, key=lambda ind: fitness(ind, parameters))[0]


def main():
    # Num. generations and pop. size tuning results (on problem 0)

    # (100, 10, np.float64(3.2031730817384574))
    # (100, 20, np.float64(1.99364723850543))
    # (100, 50, np.float64(0.01067566400550203))
    # (1000, 10, np.float64(3.3948032297143866))
    # (1000, 20, np.float64(3.2031730817384574))
    # (1000, 50, np.float64(1.8515305531317954))
    # (10000, 10, np.float64(3.8019981684757713))
    # (10000, 20, np.float64(2.1363319744909415))
    # (10000, 50, np.float64(1.8910558431660756))

    solutions: list[(Node, TrainingParameters)] = []

    for problem_num in tqdm(range(8)):
        data = np.load(f"../data/problem_{problem_num + 1}.npz")

        x: np.ndarray = data['x']
        y: np.ndarray = data['y']

        results: list[(Node, TrainingParameters)] = []

        for weighted_parent_selection in [False, True]:
            for elitist_selection in [False, True]:
                for random_event_chance in [0.0, 0.2, 0.5, 0.7, 1.0]:
                    parameters = TrainingParameters(x, y, 100, 50, weighted_parent_selection, elitist_selection, random_event_chance)

                    results.append((train_model(parameters), parameters))

        solutions.append(sorted(results, key=lambda res: DagGP.mse(res[0], x, y))[0])

    for index, solution in enumerate(solutions):
        print(f"Problem {index} \n")
        print(f"Solution: {solution[0]}, MSE: {DagGP.mse(solution[0], solution[1].x, solution[1].y)}")
        print(f"Training parameters: {solution[1]} \n\n")





if __name__ == '__main__':
    main()