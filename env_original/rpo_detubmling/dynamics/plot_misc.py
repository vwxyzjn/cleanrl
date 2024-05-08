"""
Plotting / Animation functions from the history of quaternion 
"""

import numpy as np
import matplotlib.pyplot as plt
from dynamics_rot import * 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FFMpegWriter, PillowWriter # Import FFMpegWriter
import matplotlib.animation as animation  


def plot_attitude(ax, rtn, qw, height=20):
    """
    plot attiude in 3D (just one frame)
    """
    R_i = q2rotmat(qw[0:4])
    ex, ey, ez = R_i @ np.array([height, 0, 0]), R_i @ np.array([0, height, 0]), R_i @ np.array([0, 0, height])
    # plot axis
    ax.plot3D([rtn[1], rtn[1] + ex[1]], [rtn[2], rtn[2] + ex[2]], [rtn[0], rtn[0] + ex[0]], '-r', linewidth=2)
    ax.plot3D([rtn[1], rtn[1] + ey[1]], [rtn[2], rtn[2] + ey[2]], [rtn[0], rtn[0] + ey[0]], '-g', linewidth=2)
    ax.plot3D([rtn[1], rtn[1] + ez[1]], [rtn[2], rtn[2] + ez[2]], [rtn[0], rtn[0] + ez[0]], '-b', linewidth=2)


def plot_attitude_track(ax, rtn, qw, coneAngle, height=20):
    if rtn.shape[1] != qw.shape[1]:
        raise ValueError("rtn and qw have different length. Check the input variable")

    Nfreq = rtn.shape[1] // 10  # frequency of plotting axes
    N = rtn.shape[1]

    # Cone parameters
    radius = height * np.tan(coneAngle)
    # Meshgrid for polar coordinates
    r = np.linspace(0, radius, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    r, theta = np.meshgrid(r, theta)
    # Convert polar to cartesian coordinates
    coneX = r * np.cos(theta)
    coneY = r * np.sin(theta)
    coneZ = r * height / radius  # Scale Z based on height and radius
    coneX, coneY, coneZ = coneZ, coneX, coneY  # RTN trick
    
    for i in range(N):
        if i % Nfreq == 0 or i == N-1:
            R_i = q2rotmat(qw[:4, i])
            ex, ey, ez = R_i @ np.array([height, 0, 0]), R_i @ np.array([0, height, 0]), R_i @ np.array([0, 0, height])

            # plot axis
            ax.plot3D([rtn[1, i], rtn[1, i] + ex[1]], [rtn[2, i], rtn[2, i] + ex[2]], [rtn[0, i], rtn[0, i] + ex[0]], '-r', linewidth=2)
            ax.plot3D([rtn[1, i], rtn[1, i] + ey[1]], [rtn[2, i], rtn[2, i] + ey[2]], [rtn[0, i], rtn[0, i] + ey[0]], '-g', linewidth=2)
            ax.plot3D([rtn[1, i], rtn[1, i] + ez[1]], [rtn[2, i], rtn[2, i] + ez[2]], [rtn[0, i], rtn[0, i] + ez[0]], '-b', linewidth=2)

            # plot cone 
            # coneVertices = np.vstack([coneX.flatten(), coneY.flatten(), coneZ.flatten()])
            # coneVertices = R_i @ coneVertices 
            # coneVertices = coneVertices + rtn[:3, i].reshape(-1, 1)
            # coneXRotated = coneVertices[0, :].reshape(coneX.shape)
            # coneYRotated = coneVertices[1, :].reshape(coneY.shape)
            # coneZRotated = coneVertices[2, :].reshape(coneZ.shape)
            # ax.plot_surface(coneYRotated, coneZRotated, coneXRotated, color='red', alpha=0.2, linewidth=0, antialiased=False)


def animate_attitude(rtn,qw):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    writer = PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    lines = [ax.plot([], [], [], color)[0] for color in ['r', 'g', 'b']]
    
    e_b = np.array([[1,0,0],[0,1,0],[0,0,1]])
    
    def init():
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-3, 3)
        return lines
    
    def update(i):
        R_i = q2rotmat(qw[0:4,i])
        e_i = R_i @ e_b
        for k, color in enumerate(['r', 'g', 'b']):
            lines[k].set_data([rtn[1, i], rtn[1, i] + e_i[1,k]], 
                              [rtn[2, i], rtn[2, i] + e_i[2,k]])
            lines[k].set_3d_properties([rtn[0, i], rtn[0, i] + e_i[0,k]])
            lines[k].set_color(color)
        return lines
    
    ani = animation.FuncAnimation(fig, update, rtn.shape[1], init_func=init, blit=True, interval=10)
    # ani.save('test.gif', writer=writer)
    plt.show()





