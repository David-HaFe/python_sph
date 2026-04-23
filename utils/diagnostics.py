# misc functions for stuff
# author: David Hambach Ferrer

import time
import inspect
import numpy as np
import datetime
import csv


class Diagnostics:

    def __init__(self):
        self._accepted_particles = 0
        self._rejected_particles = 0
        self._nan_instances = 0
        self._start_time = time.perf_counter()
        self._file_path = "utils/registry.csv"

        self._dynamics_timer = self._create_timer()
        self._poisson_timer = self._create_timer()
        self._kernel_timer = self._create_timer()
        self._nabla_timer = self._create_timer()
        self._laplace_timer = self._create_timer()
        self._lsqr_timer = self._create_timer()
        self._gradient_kernel_timer = self._create_timer()
        self._norm_gradient_kernel_timer = self._create_timer()
        self._surface_plot_timer = self._create_timer()
        self._position_plot_timer = self._create_timer()
        self._ode_timer = self._create_timer()
        self._logger_timer = self._create_timer()

        # set up csv file
        row = ["logged at", "name", "log content"]
        with open(self._file_path, "w+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["registered at: ", datetime.datetime.now()])
            writer.writerow(row)

    # structure:
    #   - total time spent in the function
    #   - time where last function call was started
    #   - flag stating if at the start or end of function (toggles on/off)
    #   - number of times the function has been called
    def _create_timer(self):
        return np.array([0.0, 0.0, True, 0])

    # can be called with the result of kernel rejection or acceptance to
    # get statistics for acceptance rate
    def register_particle(self, was_accepted: bool):
        if was_accepted:
            self._accepted_particles += 1
        else:
            self._rejected_particles += 1

    def register_nan(self):
        self._nan_instances += 1

    # wrapper for timing heat surface plot
    def time_surface_plot(self):
        self._timer_function(self._surface_plot_timer)

    # wrapper for timing the variable logger
    def time_logger(self):
        self._timer_function(self._logger_timer)

    # wrapper for timing particle position plot
    def time_position_plot(self):
        self._timer_function(self._position_plot_timer)

    # wrapper for timing kernel
    def time_kernel(self):
        self._timer_function(self._kernel_timer)

    # wrapper for timing nabla operator
    def time_nabla(self):
        self._timer_function(self._nabla_timer)

    # wrapper for timing laplace operator
    def time_laplace(self):
        self._timer_function(self._laplace_timer)

    # wrapper for timing the least squares estimation
    def time_least_squares(self):
        self._timer_function(self._lsqr_timer)

    # wrapper for timing gradient of W
    def time_gradient_kernel(self):
        self._timer_function(self._gradient_kernel_timer)

    # wrapper for timing laplace of W
    def time_norm_gradient_kernel(self):
        self._timer_function(self._norm_gradient_kernel_timer)

    # wrapper for timing navier stokes equations
    def time_dynamics(self):
        self._timer_function(self._dynamics_timer)

    # wrapper for timing poisson pressure equations
    def time_poisson(self):
        self._timer_function(self._poisson_timer)

    # wrapper for timing the entire ode call
    def time_ode(self):
        self._timer_function(self._ode_timer)

    # generic function able to time different parts of the program
    def _timer_function(self, timer: np.array):
        # actions at start: snapshot start time and register one call
        if timer[2]:
            timer[1] = time.perf_counter()
            timer[3] += 1
        # actions at end: snapshot end time and add to total
        else:
            timer[0] += time.perf_counter() - timer[1]

        # toggle if start or end action should be taken next
        timer[2] = not timer[2]

    # writes some basic properties of a variable to a csv file for debugging
    def log_np_array(self, array):
        self.time_logger()
        # get variable name
        array_name = ""
        frame = inspect.currentframe().f_back or inspect.currentframe()
        for name, val in list(frame.f_locals.items()):
            if val is array:
                array_name = name
                break

        dimensions = array.shape
        delta_t = time.perf_counter() - self._start_time
        time_of_registration = round(10000 * delta_t) / 10000
        row = [time_of_registration, array_name, dimensions]

        with open(self._file_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        self.time_logger()

    def print_diagnostics(self):
        print("")
        if self._accepted_particles + self._rejected_particles == 0:
            accepted_percentage = "nothing registered"
        else:
            accepted_percentage = self._accepted_particles / (
                self._accepted_particles + self._rejected_particles
            )
            accepted_percentage = f"{accepted_percentage:.4f}"

        print("========= summary ==========")
        print(" accepted percentage: " + accepted_percentage)
        print("       nan instances: " + str(self._nan_instances))
        print("========== timers ==========")
        print(
            f"                 ode: {self._ode_timer[0]:4.4f}, calls: {int(self._ode_timer[3])}"
        )
        print(
            f"            dynamics: {self._dynamics_timer[0]:4.4f}, calls: {int(self._dynamics_timer[3])}"
        )
        print(
            f"              kernel: {self._kernel_timer[0]:4.4f}, calls: {int(self._kernel_timer[3])}"
        )
        print(
            f"     kernel gradient: {self._gradient_kernel_timer[0]:4.4f}, calls: {int(self._gradient_kernel_timer[3])}"
        )
        print(
            f"norm kernel gradient: {self._norm_gradient_kernel_timer[0]:4.4f}, calls: {int(self._norm_gradient_kernel_timer[3])}"
        )
        print(
            f"       least squares: {self._lsqr_timer[0]:4.4f}, calls: {int(self._lsqr_timer[3])}"
        )
        print(
            f"               nabla: {self._nabla_timer[0]:4.4f}, calls: {int(self._nabla_timer[3])}"
        )
        print(
            f"             laplace: {self._laplace_timer[0]:4.4f}, calls: {int(self._laplace_timer[3])}"
        )
        print(
            f"        surface plot: {self._surface_plot_timer[0]:4.4f}, calls: {int(self._surface_plot_timer[3])}"
        )
        print(
            f"       position plot: {self._position_plot_timer[0]:4.4f}, calls: {int(self._position_plot_timer[3])}"
        )
        print(
            f"              logger: {self._logger_timer[0]:4.4f}, calls: {int(self._logger_timer[3])}"
        )
        print("============================")


# create diagnostics class instance to pass to other files
diagnostics = Diagnostics()
