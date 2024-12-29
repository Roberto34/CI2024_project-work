#   *        Giovanni Squillero's GP Toolbox
#  / \       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2   +      A no-nonsense GP in pure Python
#    / \
#  10   11   Distributed under MIT License

import inspect
from typing import Callable
from numpy import ufunc

__all__ = ['arity']


# Changed by myself to also work with numpy operators (e.g. numpy.add).

def arity(f: Callable | ufunc) -> int | None:
    """Return the number of expected parameter or None if variable"""
    if isinstance(f, ufunc):
        return f.nin

    try:
        return len(inspect.signature(f).parameters)
    except ValueError | TypeError:
        return None
