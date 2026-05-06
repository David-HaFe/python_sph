

import numpy as np
from config import heat_alpha

def solution(t: float, x: float, y: float):
    return (x**2 + y**2) * np.cos(t)

def source_term_heat_equation(t: float, x: float, y: float):
    return -(x**2 + y**2) * np.sin(t) - 4 * heat_alpha * np.cos(t)


