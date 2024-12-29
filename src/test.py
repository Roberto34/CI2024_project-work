from copy import deepcopy

import numpy as np

from numbers import Number
from gxgp import Node, DagGP
from src.gxgp import gxgp_random

match (1,2):
    case(0,0):
        print(0)
    case (0,1):
        print(1)
    case (1,0):
        print(2)
    case (1,1):
        print(3)
    case (2,0):
        print(4)
    case (0,2):
        print(5)
    case (2,1):
        print(6)
    case (1,2):
        print(7)
    case (2,2):
        print(8)