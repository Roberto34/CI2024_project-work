#   *        Giovanni Squillero's GP Toolbox
#  / \       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2   +      A no-nonsense GP in pure Python
#    / \
#  10   11   Distributed under MIT License

import numpy as np

from typing import Collection

from .node import Node
from .random import gxgp_random
from .utils import arity

__all__ = ['DagGP']


class DagGP:
    def __init__(self, operators: Collection, variables: int | Collection, constants: int | Collection):
        self._operators = list(operators)
        if isinstance(variables, int):
            self._variables = [Node(DagGP.default_variable(i)) for i in range(variables)]
        else:
            self._variables = [Node(t) for t in variables]
        if isinstance(constants, int):
            self._constants = [Node(gxgp_random.random()) for _ in range(constants)]
        else:
            self._constants = [Node(t) for t in constants]

    def create_individual(self, n_nodes=7) -> Node:
        pool: list[Node] = self._variables * (1 + len(self._constants) // len(self._variables)) + self._constants

        individual: Node | None = None
        while individual is None or len(individual) < n_nodes:
            op = gxgp_random.choice(self._operators)
            params: list[Node] = gxgp_random.choice(pool, size=arity(op))
            individual: Node = Node(op, params)
            pool.append(individual)
        return individual

    @property
    def operators(self):
        return list(self._operators)

    @property
    def params_pool(self):
        return self._variables * (1 + len(self._constants) // len(self._variables)) + self._constants


    @staticmethod
    def default_variable(i: int) -> str:
        return f'x{i}'

    # Note: method updated to work with numpy ndarray
    @staticmethod
    def evaluate(individual: Node, x: np.ndarray, variable_names: list[str] | None = None) -> np.ndarray:
        names: list[str]
        if variable_names:
            names = variable_names
        else:
            names = [DagGP.default_variable(i) for i in range(x.shape[0])]

        return individual(**{k: v for k, v in zip(names, x)})

    # Note: method updated to work with numpy ndarray
    @staticmethod
    def plot_evaluate(individual: Node, x: np.ndarray, variable_names: list[str] | None = None) -> np.ndarray:
        y_pred: np.ndarray = DagGP.evaluate(individual, x, variable_names)

        if x.shape == y_pred.shape:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.title(individual.long_name)
            plt.scatter(x, y_pred)

        return y_pred

    # Note: method updated to work with numpy ndarray
    @staticmethod
    def mse(individual: Node, x: np.ndarray, y: np.ndarray, variable_names: list[str] | None = None):
        y_pred: np.ndarray = DagGP.evaluate(individual, x, variable_names)
        return np.square(y - y_pred).sum() / len(y)
