import math
from config import heat_alpha


# contains the heat equation's analytical solution for validation purposes
# shift one sec to the right, because start is very awkward
def dynamics(t, x, y):
    eta = 0.01**2
    factor = 1 / (4 * math.pi * heat_alpha * (t + 1) + eta)
    exponential = math.exp(-(x**2 + y**2) / (4 * heat_alpha * (t + 1) + eta))
    return factor * exponential
