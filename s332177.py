# Copyright © 2024 Giovanni Squillero <giovanni.squillero@polito.it>
# https://github.com/squillero/computational-intelligence
# Free under certain conditions — see the license for details.

import numpy as np


def f1(x: np.ndarray) -> np.ndarray:
    # Solution: sin(x0)
    # MSE: 7.125940794232773e-34
    return np.sin(x[0])

def f2(x: np.ndarray) -> np.ndarray:
    # Original solution: multiply(625, multiply(multiply(multiply(4, multiply(multiply(4, 0.577216), multiply(4, 0.577216))), x0), multiply(multiply(4, multiply(4, 0.577216)), multiply(4, multiply(4, 0.577216)))))
    # MSE: 19019678186349.03
    # Simplified with Wolfram Alpha
    return 10240000 * x[0] * np.pow(np.euler_gamma, 4)

def f3(x: np.ndarray) -> np.ndarray:
    # Original solution: subtract(subtract(cosh(sinh(cosh(cos(cos(cosh(x0)))))), sinh(x1)), subtract(sinh(x1), cosh(x0)))
    # MSE: 188.44626602309802
    # Simplified with Wolfram Alpha
    return np.cosh(x[0]) - 2 * np.sinh(x[1]) + np.cosh(np.sinh(np.cosh(np.cos(np.cos(np.cosh(x[0]))))))

def f4(x: np.ndarray) -> np.ndarray:
    # Original solution: add(add(cos(x1), exp(cos(x1))), add(add(cos(x1), cos(cos(x1))), add(cos(x1), add(cos(x1), exp(cos(x1))))))
    # MSE: 0.41457829025162274
    # Simplified with Wolfram Alpha
    return 4 * np.cos(x[1]) + 2 * np.exp(np.cos(x[1])) + np.cos(np.cos(x[1]))

def f5(x: np.ndarray) -> np.ndarray:
    # Solution: multiply(log10(sinh(sinh(x1))), sinh(tanh(tanh(0))))
    # MSE: 5.572810232617333e-18
    # Cannot be further simplified
    return np.log10(np.sinh(np.sinh(x[1]))) * np.sinh(np.tanh(np.tanh(0)))

def f6(x: np.ndarray) -> np.ndarray:
    # Original solution: add(subtract(x1, x0), add(x1, cos(tanh(exp(subtract(subtract(x1, x0), x0))))))
    # MSE: 0.5737390617945737
    # Simplified with Wolfram Alpha
    return np.cos(np.tanh(np.exp(x[1] - 2 * x[0]))) - x[0] + 2 * x[1]

def f7(x: np.ndarray) -> np.ndarray:
    # Original solution: exp(add(multiply(x1, x0), cosh(tanh(multiply(x1, x0)))))
    # MSE: 331.07916322654233
    # Cannot be further simplified
    return np.exp(x[1] * x[0] + np.cosh(np.tanh(x[1] * x[0])))

def f8(x: np.ndarray) -> np.ndarray:
    # Problem 8: add(add(add(power(x5, 5), power(x5, 5)), add(power(x5, 5), power(x5, 5))), add(power(x5, 5), -1))
    # MSE: 735292.5511553207
    # Simplified by Wolfram Alpha
    return 5 * np.pow(x[5], 5) - 1
