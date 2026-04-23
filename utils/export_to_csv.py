# takes data and saves it
import numpy as np
import csv
from config import no_particles_x, no_particles_y, kernel_scaling
from utils.file_naming import get_file_name


def export_to_csv(t, data, file_prefix):
    filename = get_file_name(file_prefix, "solution", "csv")

    with open(filename, "w+", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "data"])
        writer.writerows(zip(t, data))
