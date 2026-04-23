import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils.diagnostics import diagnostics
from utils.file_naming import get_file_name


# this file allows to plot the positions of particles over time
def plot_particles(t, x, y, is_border_particle, file_prefix):

    diagnostics.time_position_plot()
    fig, ax = plt.subplots()
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    colors = np.where(is_border_particle, "red", "blue")
    scat = ax.scatter(x[:, 0], y[:, 0], c=colors)

    def update(frame):
        scat.set_offsets(np.c_[x[:, frame], y[:, frame]])
        ax.set_title(f"t = {t[frame]:.2f}")

        sys.stdout.write(f"\r\033[Kplotting positions @ {t[frame]}")
        sys.stdout.flush()

    ani = animation.FuncAnimation(fig, update, frames=len(t), interval=100)

    name = get_file_name(file_prefix, "position_plot", "mp4")

    ani.save(name, writer="ffmpeg", fps=30)
    diagnostics.time_position_plot()
