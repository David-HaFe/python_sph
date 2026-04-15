

from config import x_limit, y_limit, border_thickness

# takes the limits and wraps a border of a given strength around everything
def generate_border(
    r_0,
    attribute,
    p_0,
    default_attribute,
    is_border_particle,
    no_particles
):
    for x in range(0-border_thickness, x_limit+border_thickness):
        for layer in range(0, border_thickness):
            r_0.extend([x, -1-layer])
            attribute.extend(default_attribute)
            p_0.extend([1])
            is_border_particle.extend([True])

            r_0.extend([x, y_limit+layer])
            attribute.extend(default_attribute)
            p_0.extend([1])
            is_border_particle.extend([True])

            no_particles += 2

    for y in range(0, y_limit):
        for layer in range(0, border_thickness):
            r_0.extend([-1-layer, y])
            attribute.extend(default_attribute)
            p_0.extend([1])
            is_border_particle.extend([True])

            r_0.extend([x_limit+layer, y])
            attribute.extend(default_attribute)
            p_0.extend([1])
            is_border_particle.extend([True])

            no_particles += 2

    return r_0, attribute, p_0, is_border_particle, no_particles


