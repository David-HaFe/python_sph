# script to quickly compute MSE between two data sets
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils.read_from_csv import get_values_from_csv
from config import compared_files


# takes n csv files and calculates the MSE between all combinations of them
def compare_MSE():
    width = max(len(file) for file in compared_files)
    errors = np.zeros((compared_files.size, compared_files.size))

    print(" " * (width + 3), end="")
    print(" | ".join(f"{file:<{width}}" for file in compared_files))

    for index_1, file_1 in enumerate(compared_files):
        for index_2, file_2 in enumerate(compared_files):

            t_1, x_1, y_1, data1_file1, data2_file2 = get_values_from_csv(
                file_1
            )
            t_2, x_2, y_2, data1_file2, data2_file2 = get_values_from_csv(
                file_2
            )

            squared_error = (data1_file1 - data1_file2) ** 2
            mse = np.mean(squared_error)
            errors[index_1][index_2] = mse

        print()
        print(f"{file_1:<{width}} | ", end="")
        print(" | ".join(f"{error:<{width}.5f}" for error in errors[index_1]))


def compare_scatter():
    for index_1, file_1 in enumerate(compared_files):
        for index_2, file_2 in enumerate(compared_files):
            if index_1 == index_2:
                continue
            else:
                t_1, x_1, y_1, data1_file1, data2_file1 = get_values_from_csv(
                    file_1
                )
                t_2, x_2, y_2, data1_file2, data2_file2 = get_values_from_csv(
                    file_2
                )

                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")

                data_min = min(
                    np.min(data1_file1),
                    np.min(data1_file2),
                    np.min(data2_file1),
                    np.min(data2_file2),
                )

                data_max = max(
                    np.max(data1_file1),
                    np.max(data1_file2),
                    np.max(data2_file1),
                    np.max(data2_file2),
                )

                def update(frame):
                    ax.cla()
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_zlabel("data")
                    print(data_min)
                    print(data_max)
                    ax.set_zlim(data_min, data_max)
                    ax.set_title(f"t = {t_1[frame]:.2f}")
                    ax.scatter(
                        x_1[frame],
                        y_1[frame],
                        data1_file1[frame],
                        c="blue",
                        label=file_1,
                        alpha=0.5,
                        s=2,
                    )
                    ax.scatter(
                        x_2[frame],
                        y_2[frame],
                        data2_file1[frame],
                        c="red",
                        label=file_2,
                        alpha=0.5,
                        s=2,
                    )
                    ax.legend()
                    sys.stdout.write(f"\r\033[Kplotting surface @ {t_1[frame]}")
                    sys.stdout.flush()

                ani = animation.FuncAnimation(
                    fig, update, frames=len(t_1), interval=100
                )
                name = f"comparisons/{index_1}_{index_2}.mp4"
                ani.save(name, writer="ffmpeg", fps=30)
