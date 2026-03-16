

# container for particle with all its properties
class Particle():

    # constructor
    def __init__(
        position: jnp.array,
        velocity: jnp.array,
        mass: float,
        density: float,
        pressure: float,
        dynamic_viscosity: float,
    ):
        self._x = position
        self._v = velocity
        self._m = mass
        self._rho = density
        self._p = pressure
        self._mu = dynamic_viscosity

    def x():
        return self._x

    def v():
        return self._v

    def m():
        return self._m

    def rho():
        return self._rho

    def p():
        return self._p

    def mu():
        return self._mu


