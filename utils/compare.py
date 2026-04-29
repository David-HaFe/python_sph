# script to quickly compute MSE between two data sets
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import griddata
import re

from utils.read_from_npz import get_values_from_npz
from utils.diagnostics import diagnostics
from config import (
    compared_files,
    border,
    snapshots,
)


# takes n csv files and calculates the MSE between all combinations of them
def compare_MSE():
    errors = np.zeros((compared_files.size, compared_files.size))
    plot_errors = []
    plot_no_particles = []

    # derive names for table from file names
    file_names = []
    for file in compared_files:
        ana_or_num = re.search(r"analytical", file) is not None
        prefix = ana_or_num and "ana" or "num"
        numbers = re.findall(r"-?\d*\.?\d+", file)
        dimensions = f"{numbers[0]}_{numbers[1]}"
        file_names.extend([f"{prefix}_{dimensions}"])

    width = max(len(file) for file in file_names)

    print()
    print(" " * (width + 3), end="")
    print(" | ".join(f"{file_name:<{width}}" for file_name in file_names))

    grid_points = np.linspace(-border, border, 21)

    for frame in snapshots:
        for index_1, file_1 in enumerate(compared_files):
            result_1 = get_values_from_npz(file_1)
            points_1 = griddata(
                points=(result_1.x[frame, :], result_1.y[frame, :]),
                values=result_1.data_1[frame, :],
                xi=(grid_points, grid_points),
                method="cubic",
            )
            for index_2, file_2 in enumerate(compared_files):
                result_2 = get_values_from_npz(file_2)
                points_2 = griddata(
                    points=(result_2.x[frame, :], result_2.y[frame, :]),
                    values=result_2.data_1[frame, :],
                    xi=(grid_points, grid_points),
                    method="cubic",
                )
                squared_error = (points_1 - points_2) ** 2
                mse = np.mean(squared_error)
                errors[index_1][index_2] = mse

                # save errors for convergence plot
                file_1_is_analytical = re.search(r"analytical", file_1) is not None
                numbers = re.findall(r"-?\d*\.?\d+", file_1)
                numbers = [int(number) for number in numbers]
                file_1_no_particles = numbers[0]

                file_2_is_analytical = re.search(r"analytical", file_2) is not None
                numbers = re.findall(r"-?\d*\.?\d+", file_2)
                numbers = [int(number) for number in numbers]
                file_2_no_particles = numbers[0]

                one_of_each = file_1_is_analytical and not file_2_is_analytical
                same_no_of_particles = file_1_no_particles == file_2_no_particles
                if one_of_each and same_no_of_particles:
                    plot_errors.extend([mse])
                    plot_no_particles.extend([file_1_no_particles])

            # print one line of the error table
            print()
            print(f"{file_names[index_1]:<{width}} | ", end="")
            print(" | ".join(f"{error:<{width}.5f}" for error in errors[index_1]))

        # convergence plot
        plt.plot(plot_no_particles, plot_errors, "-x")

        plt.legend()
        plt.xlabel("number of particles")
        plt.ylabel("error")
        plt.title(f"errors at step {frame}")
        plt.savefig(f"comparisons/error_graph_{frame}.png")


def compare_scatter():
    diagnostics.time_scatter()

    for index_1, file_1 in enumerate(compared_files):
        for index_2, file_2 in enumerate(compared_files):
            if index_1 >= index_2:
                continue
            else:
                result_1 = get_values_from_npz(file_1)
                result_2 = get_values_from_npz(file_2)

                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")

                data_min = min(
                    np.min(result_1.data_1),
                    np.min(result_1.data_2),
                    np.min(result_2.data_1),
                    np.min(result_2.data_2),
                )

                data_max = max(
                    np.max(result_1.data_1),
                    np.max(result_1.data_2),
                    np.max(result_2.data_1),
                    np.max(result_2.data_2),
                )

                def update(frame):
                    ax.cla()
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_zlabel("data")
                    ax.set_zlim(data_min, data_max)
                    ax.set_title(f"t = {result_1.t[frame]:.2f}")
                    ax.scatter(
                        result_1.x[frame],
                        result_1.y[frame],
                        result_1.data_1[frame],
                        c="blue",
                        label=file_1,
                        alpha=0.5,
                        s=2,
                    )
                    ax.scatter(
                        result_2.x[frame],
                        result_2.y[frame],
                        result_2.data_1[frame],
                        c="red",
                        label=file_2,
                        alpha=0.5,
                        s=2,
                    )
                    ax.legend()
                    sys.stdout.write(f"\r\033[Kplotting surface @ {result_1.t[frame]}")
                    sys.stdout.flush()

                ani = animation.FuncAnimation(
                    fig, update, frames=len(result_1.t), interval=100
                )
                name = f"comparisons/{index_1}_{index_2}.mp4"
                ani.save(name, writer="ffmpeg", fps=30)

    diagnostics.time_scatter()
