

# takes data and saves it
import numpy as np
import csv
from config import no_particles_x, no_particles_y, kernel_scaling

def save_run(t, data, name):
    param = str(kernel_scaling).replace(".", "_")
    filename = f"csvs/{name}_{no_particles_x}x{no_particles_y}_r{param}.csv"

    with open(filename, "w+", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "data"])
        writer.writerows(zip(t, data))


