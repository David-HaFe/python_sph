

# file in charge of the animation
# author: David Hambach Ferrer

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_solution(t, solution):
    # solution.shape is (n_timesteps, n_particles * dims * 2)
    # unpack all timesteps at once
    no_particles = np.size(solution)//(4*len(t))
    positions_over_time = solution.reshape(-1, no_particles, 4)[:, :, :2]

    fig, ax = plt.subplots()
    ax.set_xlim(positions_over_time[:, :, 0].min() - 0.1,
                positions_over_time[:, :, 0].max() + 0.1)
    ax.set_ylim(positions_over_time[:, :, 1].min() - 0.1,
                positions_over_time[:, :, 1].max() + 0.1)
    ax.set_aspect('equal')

    # One scatter point per particle, colored individually
    scatter = ax.scatter(positions_over_time[0, :, 0],
                         positions_over_time[0, :, 1],
                         c=range(no_particles), cmap='tab10', s=100)

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def update(frame):
        scatter.set_offsets(positions_over_time[frame])
        time_text.set_text(f't = {t[frame]:.3f}')
        return scatter, time_text

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(t),
        interval=50,    # ms between frames
        blit=True,
    )

    # save to file
    ani.save('simulation.gif', writer='pillow', fps=20)

    plt.show()


