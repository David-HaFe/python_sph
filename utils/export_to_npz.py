# takes data and saves it
import numpy as np
from config import no_particles_x, no_particles_y, kernel_scaling
from utils.file_naming import get_file_name
from utils.diagnostics import diagnostics


def export_to_npz(sim_result, file_prefix):
    filename = get_file_name(file_prefix, "solution", "npz")
    diagnostics.log_string(filename)
    np.savez(
        filename,
        t=sim_result.t,
        x=sim_result.x,
        y=sim_result.y,
        data_1=sim_result.data_1,
        data_2=sim_result.data_2,
        is_border_particle=sim_result.is_border_particle,
    )
