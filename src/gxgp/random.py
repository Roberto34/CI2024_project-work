#   *        Giovanni Squillero's GP Toolbox
#  / \       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2   +      A no-nonsense GP in pure Python
#    / \
#  10   11   Distributed under MIT License

import numpy.random as random

assert "gxgp_random" not in globals(), "Paranoia check: gxgp_random already initialized"
# IMPORTANT: changed by myself to work with numpy
gxgp_random = random.default_rng(42)
