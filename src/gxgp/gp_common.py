#   *        Giovanni Squillero's GP Toolbox
#  / \       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2   +      A no-nonsense GP in pure Python
#    / \
#  10   11   Distributed under MIT License

from copy import deepcopy

from .node import Node
from .random import gxgp_random


def xover_swap_subtree(tree1: Node, tree2: Node) -> Node:
    node = None
    offspring = deepcopy(tree1)
    successors = None
    while not successors:
        node = gxgp_random.choice(list(offspring.subtree))
        successors = node.successors
    # Note: changed to reflect the changes in gxgp_random (now uses numpy.random.Generator.integers function)
    i = gxgp_random.integers(0, len(successors))
    successors[i] = deepcopy(gxgp_random.choice(list(tree2.subtree)))
    node.successors = successors
    return offspring
