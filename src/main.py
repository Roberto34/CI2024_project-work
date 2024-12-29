import copy
import re
from copy import deepcopy

import numpy as np

from tqdm import tqdm
from gxgp import Node, DagGP, gxgp_random
from src.gxgp import xover_swap_subtree, arity


class TrainingParameters:
    # Training data
    x: np.ndarray
    y: np.ndarray

    # EvAlg parameters
    num_generations: int
    population_size: int

    # Fitness parameters
    tree_depth_penalization: bool
    mse_tolerance: float

    # Crossover parameters
    weighted_parent_selection: bool
    elitist_population: bool

    # Mutation parameters
    crippling_mutation: bool
    crippling_chance: float

    def __init__(self,
            x: np.ndarray,
            y: np.ndarray,

            num_generations: int,
            population_size: int,

            tree_depth_penalization: bool = False,
            mse_tolerance: float = 0.5,

            weighted_parent_selection: bool = False,
            elitist_population: bool = False,

            crippling_mutation: bool = False,
            crippling_chance: float = 0.25,
    ):
        self.x = x
        self.y = y

        self.num_generations = num_generations
        self.population_size = population_size

        self.tree_depth_penalization = tree_depth_penalization
        self.mse_tolerance = mse_tolerance

        self.weighted_parent_selection = weighted_parent_selection
        self.elitist_population = elitist_population

        self.crippling_mutation = crippling_mutation
        self.crippling_chance = crippling_chance

class Fitness:
    individual: Node
    parameters: TrainingParameters

    def __init__(self, node: Node, parameters: TrainingParameters):
        self.individual = node
        self.parameters = parameters

    def __lt__(self, other):
        self_mse: float = DagGP.mse(self.individual, self.parameters.x, self.parameters.y)
        other_mse: float = DagGP.mse(other.individual, other.parameters.x, other.parameters.y)

        if all([re.search("^x\\d+$", sub_node.short_name) is None for sub_node in self.individual.subtree]):
            return False

        if not self.parameters.tree_depth_penalization == other.parameters.tree_depth_penalization:
            return self_mse < other_mse

        if not self.parameters.tree_depth_penalization:
            return self_mse < other_mse

        if not self.parameters.mse_tolerance == other.parameters.mse_tolerance:
            return self_mse < other_mse

        if abs(self_mse - other_mse) > self.parameters.mse_tolerance:
            return self_mse < other_mse

        return len(self.individual.subtree) < len(other.individual.subtree)

    @property
    def mse(self):
        return DagGP.mse(self.individual, self.parameters.x, self.parameters.y)


def cleanup(individual: Node) -> Node:
    if any([(not node.is_leaf and not is_cleaned_up(node)) for node in individual.successors]):
        individual.successors = [cleanup(successor) for successor in individual.successors]

        return cleanup(individual)

    if individual.short_name == "add":
        non_zero_nodes: list[Node] = list(filter(lambda node: node.long_name != "0", individual.successors))

        if len(non_zero_nodes) == 0:
            return Node(0)
        if len(non_zero_nodes) == 1:
            return non_zero_nodes[0]

    if individual.short_name == "subtract":
        if individual.successors[1].long_name == "0":
            return individual.successors[0]

        if individual.successors[0].long_name == individual.successors[1].long_name:
            return Node(0)

    if individual.short_name == "multiply":
        non_zero_nodes: list[Node] = list(filter(lambda node: node.long_name != "0", individual.successors))
        non_one_nodes: list[Node] = list(filter(lambda node: node.long_name != "1", individual.successors))

        if len(non_zero_nodes) != 2:
            return Node(0)

        if len(non_one_nodes) == 0:
            return Node(1)
        if len(non_one_nodes) == 1:
            return non_one_nodes[0]

    if individual.short_name == "divide":
        if individual.successors[0].long_name == "0":
            return Node(0)

        if individual.successors[1].long_name == "1":
            return individual.successors[0]

    return individual

def is_cleaned_up(individual: Node) -> bool:
    for sub_node in individual.subtree:
        if sub_node.short_name == "add":
            non_zero_nodes: list[Node] = list(filter(lambda node: node.long_name != "0", sub_node.successors))

            if len(non_zero_nodes) != 2:
                return False

        if sub_node.short_name == "subtract":
            if sub_node.successors[1].long_name == "0" or sub_node.successors[0].long_name == sub_node.successors[1].long_name:
                return False

        if individual.short_name == "multiply":
            non_zero_nodes: list[Node] = list(filter(lambda node: node.long_name != "0", individual.successors))
            non_one_nodes: list[Node] = list(filter(lambda node: node.long_name != "1", individual.successors))

            if len(non_zero_nodes) != 2 or len(non_one_nodes) != 2:
                return False

        if individual.short_name == "divide":
            if individual.successors[0].long_name == "0" or individual.successors[1].long_name == "1":
                return False

    return True


def fitness(individual: Node, parameters: TrainingParameters) -> Fitness:
    return Fitness(individual, parameters)

def select_parents(parent_pool: list[Node], parameters: TrainingParameters) -> (Node, Node):
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

        if successors[index].is_leaf or (parameters.crippling_mutation and gxgp_random.random() < parameters.crippling_chance):
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
            np.sin, np.cos,
            np.exp, np.log
        ],
        parameters.x.shape[0],
        [-1, 0, 1, 2, 5, np.pi, np.e])

    # Create an initial population of size N
    population: list[Node] = [cleanup(dag.create_individual()) for _ in range(parameters.population_size)]

    # If any of the initial solutions already has an MSE value of 0.0 (extremely unlikely, but possible), returns
    # that solution and skips the training completely.
    final_solution: Node | None = check_population(population, parameters)
    if final_solution is not None:
        return final_solution

    for i in tqdm(range(parameters.num_generations)):
        # Sort the population by fitness
        sorted_population: list[Node] = sorted(population, key=lambda ind: fitness(ind, parameters))

        if verbose:
            print(f"Generation {i}: completed sorting")

        # Restrict the pool of available parents to the best N / 2 individuals
        parents_pool = sorted_population[:parameters.population_size // 2]

        if verbose:
            print(f"Generation {i}: generated parents pool")

        # Create a new generation
        population = []
        # Fill half the population of this generation with children generated by crossover
        for _ in range(parameters.population_size // 2):
            parents: tuple[Node, Node] = select_parents(parents_pool, parameters)
            population.append(crossover(parents))

        if verbose:
            print(f"Generation {i}: completed crossover")

        # For the other half, if an elitist strategy is being applied, pick the best individuals from the previous generation
        if parameters.elitist_population:
            population.extend(sorted_population[:parameters.population_size // 2])
        # otherwise, generate children like before
        else:
            for _ in range(parameters.population_size // 2):
                parents: tuple[Node, Node] = select_parents(parents_pool, parameters)
                population.append(crossover(parents))

        if verbose:
            print(f"Generation {i}: created new generation")

        # Check for the (extremely unlikely) case where a solution has MSE of 0.0.
        # If it does, stop the computations and return that solution.
        final_solution: Node | None = check_population(population, parameters)
        if final_solution is not None:
            return final_solution

        population = [mutate(individual, dag, parameters) for individual in population]

        # Check again for the stopping case.
        final_solution: Node | None = check_population(population, parameters)
        if final_solution is not None:
            return final_solution

        # Clean up the population (i.e. simplify the expression when possible)
        population = [cleanup(individual) for individual in population]
        if verbose:
            print(f"Generation {i}: completed cleanup")

    return sorted(population, key=lambda ind: fitness(ind, parameters))[0]




def main():
    data = np.load("../data/problem_0.npz")

    x: np.ndarray = data['x']
    y: np.ndarray = data['y']

    training_params: TrainingParameters = TrainingParameters(
        x, y,
        10,
        20,
        tree_depth_penalization=True,
        #weighted_parent_selection=True,
        #elitist_population=True,
        crippling_mutation=True
    )

    print(train_model(training_params))










if __name__ == '__main__':
    main()