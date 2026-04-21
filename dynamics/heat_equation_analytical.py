import math


# contains the heat equation's analytical solution for validation purposes
def heat_equation_analytical(t, x, y):
    alpha = 0.2
    return (
        (1 / (4 * math.pi * alpha * t))
        * math.exp(-(x**2 + y**2) / (4 * alpha * t))
    )
