

import numpy as np

from config import (
    no_particles_x,
    no_particles_y,
    border_thickness,
    border,
    dx,
    dy,
)

# takes the limits and wraps a border of a given strength around everything
def generate_border(
    r_0,
    attribute,
    p_0,
    default_attribute,
    is_border_particle,
):
    x_positions = np.linspace(
        - border - border_thickness*dx,
        + border + border_thickness*dx,
        no_particles_x + 2*border_thickness,
    )
    y_positions = np.linspace(dy, border_thickness*dy, border_thickness)
    for _, x in enumerate(x_positions):
        for _, y in enumerate(y_positions):
            r_0.extend([x, -border-y])
            attribute.extend(default_attribute)
            p_0.extend([1])
            is_border_particle.extend([True])

            r_0.extend([x, border+y])
            attribute.extend(default_attribute)
            p_0.extend([1])
            is_border_particle.extend([True])

    x_positions = np.linspace(dx, border_thickness*dx, border_thickness)
    y_positions = np.linspace(
        - border,
        + border,
        no_particles_y,
    )
    for _, y in enumerate(y_positions):
        for _, x in enumerate(x_positions):
            r_0.extend([-border-x, y])
            attribute.extend(default_attribute)
            p_0.extend([1])
            is_border_particle.extend([True])

            r_0.extend([border+x, y])
            attribute.extend(default_attribute)
            p_0.extend([1])
            is_border_particle.extend([True])

    return r_0, attribute, p_0, is_border_particle


