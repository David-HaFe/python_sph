

# misc functions for stuff
# author: David Hambach Ferrer

class Diagnostics():

    def __init__(self):
        self._accepted_particles = 0
        self._rejected_particles = 0

    # can be called with the result of kernel rejection or acceptance to
    # get statistics for acceptance rate
    def register_particle(
            self,
            was_accepted: bool,
    ):
        if was_accepted:
            self._accepted_particles += 1
        else:
            self._rejected_particles += 1

    def print_statistics(self):
        accepted_percentage = self._accepted_particles/(
                self._accepted_particles + self._rejected_particles)
        print("accepted percentage: " + str(accepted_percentage))

# create diagnostics class instance to pass to other files
diagnostics = Diagnostics()


