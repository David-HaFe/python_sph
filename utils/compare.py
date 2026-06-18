# script to quickly compute MSE between two data sets
import numpy as np
import pandas as pd
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import ScalarFormatter, FuncFormatter, NullFormatter
from scipy.interpolate import griddata
import re

from utils.read_from_npz import get_values_from_npz
from utils.diagnostics import diagnostics
from config import (
    sim_run_parameters,
    compared_files,
    border,
    snapshots,
)

# font size
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

# # Thicker axis frame
# for spine in plt.gca().spines.values():
#     spine.set_linewidth(2)

# Thicker tick marks
plt.tick_params(axis='both', width=2, length=6)

# takes n csv files and calculates the MSE between all combinations of them
def compare_MSE():
    errors = np.zeros((compared_files.size, compared_files.size))

    # derive names for table from file names
    file_names = []
    for file in compared_files:
        is_ana = re.search(r"analytical", file) is not None
        is_man = re.search(r"manufactured", file) is not None
        prefix = is_ana and "ana" or is_man and "man" or "num"
        numbers = re.findall(r"-?\d*\.?\d+", file)
        dimensions = f"{numbers[0]}x{numbers[1]}"
        radius = f"{numbers[2]}.{numbers[3]}"
        steps = f"{numbers[4]}.{numbers[5]}"

        file_names.extend([f"{prefix} {dimensions} r{radius} dt{steps}"])

    width = max(len(file) for file in file_names)

    grid_points = np.linspace(-border, border, 21)

    for second in snapshots:
        print()
        print(f"run for step {second} ---------------------------")
        print(" " * (width + 3), end="")
        print(" | ".join(f"{file_name:<{width}}" for file_name in file_names))

        plot_errors = []
        plot_no_particles = []
        plot_kernel_radius = []
        plot_step_size = []
        mse = []

        for index_1, file_1 in enumerate(compared_files):
            result_1 = get_values_from_npz(file_1)
            params_1 = _get_information_from_name(file_1)
            frame_1 = int(second / params_1.step_size)
            points_1 = griddata(
                points=(result_1.x[frame_1, :], result_1.y[frame_1, :]),
                values=result_1.data_1[frame_1, :],
                xi=(grid_points, grid_points),
                method="cubic",
            )
            for index_2, file_2 in enumerate(compared_files):
                result_2 = get_values_from_npz(file_2)
                params_2 = _get_information_from_name(file_2)
                frame_2 = int(second / params_2.step_size)
                points_2 = griddata(
                    points=(result_2.x[frame_2, :], result_2.y[frame_2, :]),
                    values=result_2.data_1[frame_2, :],
                    xi=(grid_points, grid_points),
                    method="cubic",
                )
                squared_error = (points_1 - points_2) ** 2
                mse = np.mean(squared_error)
                errors[index_1][index_2] = mse

                # save errors for convergence plot
                analytical_match = params_1.is_analytical and not params_2.is_analytical
                manufactured_match = (
                    params_1.is_manufactured and not params_2.is_manufactured
                )
                same_no_of_particles = params_1.no_particles == params_2.no_particles
                same_step_size = params_1.step_size == params_2.step_size
                same_radius = params_1.kernel_radius == params_2.kernel_radius
                same_params = same_no_of_particles and same_step_size and same_radius

                if (analytical_match or manufactured_match) and same_params:
                    plot_errors.extend([mse])
                    plot_no_particles.extend([params_1.no_particles])
                    plot_kernel_radius.extend([params_1.kernel_radius])
                    plot_step_size.extend([params_1.step_size])

            # print one line of the error table
            print()
            print(f"{file_names[index_1]:<{width}} | ", end="")
            print(" | ".join(f"{error:<{width}.5f}" for error in errors[index_1]))

        # convergence plot
        compared_parameter = "radius"

        plt.clf()
        plt.grid()
        plt.ylabel("error")
        plt.title(f"errors at {second} s")

        def x_log_formatter(x, pos):
            return f"{x:g}"
        def y_log_formatter(x, pos):
            return f"{x:.2f}"
        plt.minorticks_off()

        match compared_parameter:
            case "step":
                plt.xlabel("step size")
                plt.loglog(plot_step_size, plot_errors, "-x")
                plt.gca().xaxis.set_minor_formatter(NullFormatter())
                plt.gca().xaxis.set_major_formatter(FuncFormatter(x_log_formatter))
                # plt.gca().yaxis.set_major_formatter(FuncFormatter(y_log_formatter))
                plt.xticks([0.0125, 0.025, 0.05, 0.1])
                plt.tight_layout()
                plt.savefig(f"comparisons/error_step_{second}.png", dpi=300)
            case "grid":
                plt.xlabel("grid dimension")
                plt.loglog(plot_no_particles, plot_errors, "-x")
                plt.gca().xaxis.set_minor_formatter(NullFormatter())
                plt.gca().xaxis.set_major_formatter(FuncFormatter(x_log_formatter))
                # plt.gca().yaxis.set_major_formatter(FuncFormatter(y_log_formatter))
                plt.xticks([6, 11, 21])
                plt.tight_layout()
                plt.savefig(f"comparisons/error_grid_{second}.png", dpi=300)
            case "radius":
                plt.xlabel("relative kernel radius")
                plt.loglog(plot_kernel_radius, plot_errors, "-x")
                plt.gca().xaxis.set_minor_formatter(NullFormatter())
                plt.gca().xaxis.set_major_formatter(FuncFormatter(x_log_formatter))
                # plt.gca().yaxis.set_major_formatter(FuncFormatter(y_log_formatter))
                plt.xticks([1.5, 2, 3, 4.5])
                plt.tight_layout()
                plt.savefig(f"comparisons/error_radius_{second}.png", dpi=300)

        plt.close()


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
                    ax.set_zlabel("T")
                    ax.set_zlim(data_min, data_max)
                    ax.set_title(f"t = {result_1.t[frame]:.2f}")
                    ax.scatter(
                        result_1.x[frame],
                        result_1.y[frame],
                        result_1.data_1[frame],
                        c="blue",
                        label="numerical",
                        alpha=0.5,
                        s=2,
                    )
                    ax.scatter(
                        result_2.x[frame],
                        result_2.y[frame],
                        result_2.data_1[frame],
                        c="red",
                        label="manufactured",
                        alpha=0.5,
                        s=2,
                    )
                    ax.legend()
                    sys.stdout.write(f"\r\033[Kplotting scatter @ {result_1.t[frame]}")
                    sys.stdout.flush()

                ani = animation.FuncAnimation(
                    fig, update, frames=len(result_1.t), interval=100
                )
                name = f"comparisons/{index_1}_{index_2}.mp4"
                ani.save(name, writer="ffmpeg", fps=30, dpi=300)

    diagnostics.time_scatter()


# can be called to get all the information one needs about the file
def _get_information_from_name(filename):
    is_analytical = re.search(r"analytical", filename) is not None
    is_manufactured = re.search(r"manufactured", filename) is not None

    numbers = re.findall(r"-?\d*\.?\d+", filename)
    numbers = [int(number) for number in numbers]
    no_particles = numbers[0]

    kernel_radius = re.search(r"_r(\d+)_(\d+)", filename)
    kernel_radius = float(f"{kernel_radius.group(1)}.{kernel_radius.group(2)}")

    step_size = re.search(r"_dt(\d+)_(\d+)", filename)
    step_size = float(f"{step_size.group(1)}.{step_size.group(2)}")

    result = sim_run_parameters(
        is_analytical=is_analytical,
        is_manufactured=is_manufactured,
        no_particles=no_particles,
        kernel_radius=kernel_radius,
        step_size=step_size,
    )
    return result
