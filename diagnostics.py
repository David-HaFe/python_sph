

# misc functions for stuff
# author: David Hambach Ferrer

import time
import numpy as np

class Diagnostics():

    def __init__(self):
        self._accepted_particles = 0
        self._rejected_particles = 0
        self._nan_instances = 0

        self._navier_stokes_timer = np.array([0.0, 0.0, True])
        self._W_timer = np.array([0.0, 0.0, True])
        self._delta_W_timer = np.array([0.0, 0.0, True])
        self._ode_timer = np.array([0.0, 0.0, True])

    # can be called with the result of kernel rejection or acceptance to
    # get statistics for acceptance rate
    def register_particle(self, was_accepted: bool):
        if was_accepted:
            self._accepted_particles += 1
        else:
            self._rejected_particles += 1

    def register_nan(self):
        self._nan_instances += 1

    # wrapper for timing W
    def time_W(self):
        self._timer_function(self._W_timer)

    # wrapper for timing delta W
    def time_delta_W(self):
        self._timer_function(self._delta_W_timer)

    # wrapper for timing navier stokes equations
    def time_navier_stokes(self):
        self._timer_function(self._navier_stokes_timer)

    # wrapper for timing the entire ode call
    def time_ode(self):
        self._timer_function(self._ode_timer)

    # generic function able to time different parts of the program
    def _timer_function(self, timer: np.array):
        # actions at start: snapshot start time
        if timer[2]:
            timer[1] = time.perf_counter()
        # actions at end: snapshot end time and add to total
        else:
            timer[0] += time.perf_counter() - timer[1]

        # toggle if start or end action should be taken next
        timer[2] = not timer[2]

    def print_diagnostics(self):
        accepted_percentage = self._accepted_particles/(
                self._accepted_particles + self._rejected_particles)
        print("accepted percentage: " + str(accepted_percentage))
        print("      nan instances: " + str(self._nan_instances))
        print("navier stokes time : " + str(self._navier_stokes_timer[0]))
        print("            W time : " + str(self._W_timer[0]))
        print("      delta W time : " + str(self._delta_W_timer[0]))
        print("           ode time: " + str(self._ode_timer[0]))

# create diagnostics class instance to pass to other files
diagnostics = Diagnostics()


