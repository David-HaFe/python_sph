import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.interpolate import griddata
from utils.diagnostics import diagnostics
from utils.file_naming import get_file_name

from config import (
    no_particles_x,
    no_particles_y,
    kernel_scaling,
)


def plot_temperature_surface(t, x, y, T, file_prefix):
    diagnostics.time_surface_plot()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    grid_points = 50

    # xi, yi don't change, just calculate once and continue
    xi = np.linspace(x[:, 1].min(), x[:, 1].max(), grid_points)
    yi = np.linspace(y[:, 1].min(), y[:, 1].max(), grid_points)
    XI, YI = np.meshgrid(xi, yi)
    z_min = np.nanmin(T)
    z_max = np.nanmax(T)

    def update(frame):
        ax.cla()
        ax.set_zlim(z_min, z_max * 2)
        ax.set_title(f"t = {t[frame]:.2f}")
        ZI = griddata(
            (x[:, frame], y[:, frame]),
            T[:, frame],
            (XI, YI),
            method="cubic",
        )
        ax.plot_surface(
            XI,
            YI,
            ZI,
            cmap="viridis",
            edgecolor="none",
            vmax=5,
            vmin=-0.1,
        )
        sys.stdout.write(f"\r\033[Kplotting surface @ {t[frame]}")
        sys.stdout.flush()

    ani = animation.FuncAnimation(fig, update, frames=len(t), interval=100)

    name = get_file_name(file_prefix, "heat_surface", "mp4")

    ani.save(name, writer="ffmpeg", fps=30)
    diagnostics.time_surface_plot()


def plot_temperature_map(sim_result, file_prefix):
    diagnostics.time_surface_plot()

    fig, ax = plt.subplots()
    grid_points = 50

    xi = np.linspace(sim_result.x[1, :].min(), sim_result.x[1, :].max(), grid_points)
    yi = np.linspace(sim_result.y[1, :].min(), sim_result.y[1, :].max(), grid_points)
    XI, YI = np.meshgrid(xi, yi)

    diagnostics.log_string(f"evaulating {file_prefix}")
    diagnostics.log_full_np_array(sim_result.x)
    diagnostics.log_full_np_array(sim_result.y)
    diagnostics.log_full_np_array(sim_result.data_1)
    diagnostics.log_full_np_array(sim_result.data_2)

    def update(frame):
        ax.cla()
        ax.set_title(f"t = {sim_result.t[frame]:.2f}")
        ZI = griddata(
            (sim_result.x[frame, :], sim_result.y[frame, :]),
            sim_result.data_1[frame, :],
            (XI, YI),
            method="cubic",
        )
        heatmap = ax.imshow(
            ZI,
            extent=[xi.min(), xi.max(), yi.min(), yi.max()],
            origin="lower",
            cmap="viridis",
            vmax=1,
            vmin=-0.1,
            aspect="auto",
        )
        sys.stdout.write(f"\r\033[Kplotting heatmap @ {sim_result.t[frame]}")
        sys.stdout.flush()

    ani = animation.FuncAnimation(fig, update, frames=len(sim_result.t), interval=100)

    name = get_file_name(file_prefix, "heat_map", "mp4")

    ani.save(name, writer="ffmpeg", fps=30)
    diagnostics.time_surface_plot()
