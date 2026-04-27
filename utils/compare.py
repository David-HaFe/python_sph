# script to quickly compute MSE between two data sets
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import griddata

from utils.read_from_npz import get_values_from_npz
from utils.diagnostics import diagnostics
from config import (
    compared_files,
    border,
)

# takes n csv files and calculates the MSE between all combinations of them
def compare_MSE():
    width = max(len(file) for file in compared_files)
    errors = np.zeros((compared_files.size, compared_files.size))

    print(" " * (width + 3), end="")
    print(" | ".join(f"{file:<{width}}" for file in compared_files))

    grid_points = np.linspace(-border, border, 21)

    for index_1, file_1 in enumerate(compared_files):
        for index_2, file_2 in enumerate(compared_files):
            for frame, _ in enumerate(result_1.t):

                result_1 = get_values_from_npz(file_1)
                result_2 = get_values_from_npz(file_2)

                points_1 = griddata(
                    points=(result_1.x[frame, :], result_1.y[frame, :]),
                    values=result_1.data_1[frame, :],
                    xi=(grid_points, grid_points),
                    method='cubic',
                )

                points_2 = griddata(
                    points=(result_1.x[frame, :], result_1.y[frame, :]),
                    values=result_1.data_1[frame, :],
                    xi=(grid_points, grid_points),
                    method='cubic',
                )

            squared_error = (result_1.data_1 - result_2.data_1) ** 2
            mse = np.mean(squared_error)
            errors[index_1][index_2] = mse

        print()
        print(f"{file_1:<{width}} | ", end="")
        print(" | ".join(f"{error:<{width}.5f}" for error in errors[index_1]))


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
