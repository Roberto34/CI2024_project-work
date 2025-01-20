import warnings
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

    # Fitness parameters
    mse_percentage_tolerance: float

    # Chance parameters
    random_event_chance: float

    def __init__(self,
            x: np.ndarray,
            y: np.ndarray,

            num_generations: int,
            population_size: int,

            mse_percentage_tolerance: float = 0.0,

            random_event_chance: float = 0.0,
    ):
        self.x = x
        self.y = y

        self.num_generations = num_generations
        self.population_size = population_size

        self.mse_percentage_tolerance = mse_percentage_tolerance

        self.random_event_chance = random_event_chance

    def __str__(self):
        return (f"MSE tolerance: {self.mse_percentage_tolerance} % \n"
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

        # If the mse is only bigger up to a margin (defined by the hyperparameter), compare the individuals based on subtree length.
        if self_mse > other_mse and self_mse / other_mse - 1 <= self.parameters.mse_percentage_tolerance:
            return len(self.individual.subtree) < len(other.individual.subtree)

        return self_mse < other_mse


    @property
    def mse(self):
        return DagGP.mse(self.individual, self.parameters.x, self.parameters.y)


# An individual is defined as valid if it has at least a variable node, it can be evaluated, and it has a subtree length < 15.
def is_valid(individual: Node, parameters: TrainingParameters) -> bool:
    try:
        return (
            any(sub_node.is_variable for sub_node in individual.subtree) and
            not np.any(np.isnan(DagGP.evaluate(individual, parameters.x))) and
            not np.any(np.isinf(DagGP.evaluate(individual, parameters.x))) and
            len(individual.subtree) <= 25
        )
    except BaseException:
        return False


# Removes unnecessary nodes, appropriately replacing them with cleaned-up versions (e.g. replaces add(0, x0) with just x0
def cleanup(individual: Node) -> Node:
    try:
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
    except BaseException:
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
    return tuple(gxgp_random.choice(parent_pool, size=2, replace=False))


# Generates an offspring from 2 parents by inserting a subtree of the 2nd parent in the 1st.
# The subtree is chosen to be of length > 1, so the length of the offspring will generally be longer than the parents (excluding errors)
def crossover(parents: (Node, Node)) -> Node:
    try:
        offspring: Node = deepcopy(parents[0])

        successors: list[Node] | None = None
        pivot: Node | None = None

        for _ in range(100):
            pivot = gxgp_random.choice(list(offspring.subtree))
            successors = pivot.successors

            if len(successors) > 0:
                break

        if len(successors) == 0:
            return gxgp_random.choice(parents)

        new_subtree: Node | None = None

        for _ in range(100):
            new_subtree = gxgp_random.choice(list(parents[1].subtree))

            if len(new_subtree.successors) > 0:
                break

        if len(new_subtree.successors) == 0:
            return gxgp_random.choice(parents)

        successors[gxgp_random.integers(len(successors))] = new_subtree
        pivot.successors = successors

        return offspring
    except BaseException:
        return gxgp_random.choice(parents)


# Randomly replaces half of an individual's nodes with nodes of the same type (leaves or operators with the same arity)
# ALso has a chance to "cut" a tree, replacing an operator with a leaf. This operation is also applied when the individual
# is at max length (with a 1 in 2 chance), to avoid saturation.
def mutate(individual: Node, dag: DagGP, parameters: TrainingParameters) -> Node:
    try:
        mutated_individual: Node = deepcopy(individual)

        for _ in range(len(mutated_individual.subtree) // 2):
            pivot: Node | None = None
            successors: list[Node] | None = None

            for _ in range(100):
                pivot = gxgp_random.choice(list(mutated_individual.subtree))
                successors = pivot.successors

                if len(successors) > 0:
                    break

            if len(successors) == 0:
                return mutated_individual

            index: int = gxgp_random.integers(len(successors))

            if successors[index].is_leaf or gxgp_random.random() < parameters.random_event_chance:
                successors[index] = gxgp_random.choice(dag.params_pool)
            else:
                chosen_operator: np.ufunc = gxgp_random.choice(dag.operators)
                operator_successors: list[Node] = list(gxgp_random.choice(successors[index].successors, size=successors[index].arity))

                if len(operator_successors) < successors[index].arity:
                    operator_successors.append(gxgp_random.choice(dag.params_pool, size = successors[index].arity - len(operator_successors)))

                successors[index] = Node(chosen_operator, operator_successors)

            pivot.successors = successors

        return mutated_individual
    except BaseException:
        return individual


# Checks for the (extremely unlikely) case in which an individual already has an MSE of 0.
def check_population(population: list[Node], parameters: TrainingParameters) -> Node | None:
    for individual in population:
        if DagGP.mse(individual, parameters.x, parameters.y) == 0:
            return individual

    return None


# Trains a model with the specified hyperparameters and returns the best solution.
def train_model(parameters: TrainingParameters, verbose: bool = False) -> Node:
    dag: DagGP = DagGP(
        [
            np.add, np.subtract,
            np.multiply, np.divide,
            np.pow, np.abs,
            np.sin, np.sinh,
            np.cos, np.cosh,
            np.tan, np.tanh,
            np.exp, np.log,
            np.log2, np.log10
        ],
        parameters.x.shape[0],
        [-1, 0, 1, 2, 5, np.pi, np.e, np.euler_gamma]
    )

    # Create an initial population of size N
    population: list[Node] = []
    sorted_population: list[Node] = []
    for _ in range(parameters.population_size * 50):
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

    for i in tqdm(range(parameters.num_generations)):
        # Sort the population by fitness
        sorted_population.clear()
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
        population.clear()
        # Fill the population of this generation with children generated by crossover
        for _ in range(parameters.population_size * 50):
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
                    break

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
    pass

    # Num. generations and pop. size tuning results (on problem 0)

    #data = np.load("../data/problem_0.npz")
    #x, y = data["x"], data["y"]
    #results: list = []

    #for num_generations in [100, 1000, 10000]:
    #    for population_size in [100, 20, 50]:
    #        with warnings.catch_warnings():
    #            warnings.simplefilter("ignore")
    #            try:
    #                results.append((num_generations, population_size, DagGP.mse(train_model(TrainingParameters(x, y, num_generations, population_size)), x, y)))
    #            except Exception:
    #                results.append((num_generations, population_size, np.inf))

    #for _ in range(10):
    #    for num_generations in [100, 500, 1000]:
    #        with warnings.catch_warnings():
    #            warnings.simplefilter("ignore")
    #            try:
    #                results.append((num_generations, DagGP.mse(train_model(TrainingParameters(x, y, num_generations, 50)), x, y)))
    #            except Exception:
    #                results.append((num_generations, np.inf))

    #or result in results:
    #   print(result)

    # (100, 10, np.float64(3.2031730817384574))
    # (100, 20, np.float64(1.99364723850543))
    # (100, 50, np.float64(0.01067566400550203))

    # (1000, 10, np.float64(3.3948032297143866))
    # (1000, 20, np.float64(3.2031730817384574))
    # (1000, 50, np.float64(1.8515305531317954))

    # (10000, 10, np.float64(3.8019981684757713))
    # (10000, 20, np.float64(2.1363319744909415))
    # (10000, 50, np.float64(1.8910558431660756))


    # (100, np.float64(0.01067566400550203))
    # (500, np.float64(0.01067566400550203))
    # (1000, np.float64(0.01067566400550203))

    # (100, np.float64(2.2034576748476447e-05))
    # (500, np.float64(0.01067566400550203))
    # (1000, np.float64(0.009048410963202195))

    # (100, np.float64(0.01067566400550203))
    # (500, np.float64(0.01067566400550203))
    # (1000, np.float64(0.01067566400550203))

    # (100, np.float64(0.01067566400550203))
    # (500, np.float64(0.01067566400550203))
    # (1000, np.float64(0.01067566400550203))

    # (100, np.float64(0.01067566400550203))
    # (500, np.float64(0.01067566400550203))
    # (1000, np.float64(0.01067533530166119))

    # (100, np.float64(0.0013864402896065096))
    # (500, np.float64(0.010643207535721956))
    # (1000, np.float64(0.010627995865371602))

    # (100, np.float64(0.03245070175135962))
    # (500, np.float64(2.3699041605483037e-05))
    # (1000, np.float64(1.4651593566170996e-07))

    # (100, np.float64(0.010708283456020444))
    # (500, np.float64(0.010675468914254126))
    # (1000, np.float64(0.01067566400550203))

    # (100, np.float64(0.010675309304174842))
    # (500, np.float64(0.0015832048376789383))
    # (1000, np.float64(0.25434601496202247))

    # (100, np.float64(0.01067566400550203))
    # (500, np.float64(0.010540566792826478))
    # (1000, np.float64(0.01067566400550203))

    # Hyperparameter tuning

    # data = np.load(f"../data/problem_8.npz")
    # x, y = data["x"], data["y"]
    # results: list[tuple[Node, TrainingParameters]] = []

    # for weighted_parent_selection in [False, True]:
    #     for mse_tolerance in [0.0, 0.05, 0.1]:
    #         for random_event_chance in [0.0, 0.25, 0.5, 0.75, 1.0]:
    #             parameters = TrainingParameters(x, y, 500, 50, mse_tolerance, weighted_parent_selection, random_event_chance)

    #             with warnings.catch_warnings():
    #                 warnings.simplefilter("ignore")
    #                 try:
    #                     results.append((train_model(parameters), parameters))
    #                 except BaseException:
    #                     pass

    # best_result: tuple[Node, TrainingParameters] = sorted(results, key=lambda res: DagGP.mse(res[0], res[1].x, res[1].y))[0]

    # print(f"Best parameters for problem 8: \n")
    # print(best_result[0], DagGP.mse(best_result[0], x, y))
    # print(best_result[1])

    # Problem 1

    # MSE tolerance: 0.0
    # Random event chance: 0.75

    # Problem 2

    # MSE tolerance: 0.0
    # Random event chance: 0.25

    # Problem 3

    # MSE tolerance: 0.1
    # Random event chance: 0.0

    # Problem 4

    # MSE tolerance: 0.0
    # Random event chance: 0.25

    # Problem 5

    # MSE tolerance: 0.1
    # Random event chance: 0.0

    # Problem 6

    # MSE tolerance: 0.05
    # Random event chance: 0.25

    # Problem 7

    # MSE tolerance: 0.05
    # Random event chance: 0.0

    # Problem 8

    # MSE tolerance: 0.05
    # Random event chance: 0.5

    #data = np.load("../data/problem_8.npz")
    #x, y = data["x"], data["y"]
    #parameters = TrainingParameters(x, y, 500, 50, 0.05, 0.5)

    #solutions: list[Node] = []

    #for _ in range(10):
    #    with warnings.catch_warnings():
    #        warnings.simplefilter("ignore")

    #        try:
    #            solutions.append(train_model(parameters))
    #        except Exception:
    #            pass

    #best_solution: Node = sorted(solutions, key=lambda ind: DagGP.mse(ind, x, y))[0]

    #print(f"Solution: {best_solution}, MSE: {DagGP.mse(best_solution, x, y)}")

    # Final solutions:

    # Problem 1: sin(x0), MSE: 7.125940794232773e-34
    # Problem 2: multiply(625, multiply(multiply(multiply(4, multiply(multiply(4, 0.577216), multiply(4, 0.577216))), x0), multiply(multiply(4, multiply(4, 0.577216)), multiply(4, multiply(4, 0.577216))))), MSE: 19019678186349.03
    # Problem 3: subtract(subtract(cosh(sinh(cosh(cos(cos(cosh(x0)))))), sinh(x1)), subtract(sinh(x1), cosh(x0))), MSE: 188.44626602309802
    # Problem 4: add(add(cos(x1), exp(cos(x1))), add(add(cos(x1), cos(cos(x1))), add(cos(x1), add(cos(x1), exp(cos(x1)))))), MSE: 0.41457829025162274
    # Problem 5: multiply(log10(sinh(sinh(x1))), sinh(tanh(tanh(0)))), MSE: 5.572810232617333e-18
    # Problem 6: add(subtract(x1, x0), add(x1, cos(tanh(exp(subtract(subtract(x1, x0), x0)))))), MSE: 0.5737390617945737
    # Problem 7: exp(add(multiply(x1, x0), cosh(tanh(multiply(x1, x0))))), MSE: 331.07916322654233
    # Problem 8: add(add(add(power(x5, 5), power(x5, 5)), add(power(x5, 5), power(x5, 5))), add(power(x5, 5), -1)), MSE: 735292.5511553207


if __name__ == '__main__':
    main()