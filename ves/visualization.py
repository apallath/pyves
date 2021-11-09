"""
Classes for visualizing trajectory data.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from potentials import Potential2D

# Resource-light and SSH-friendly non-GUI plotting
matplotlib.use('Agg')


class VisualizePotential2D:
    def __init__(self,
                 potential2D: Potential2D,
                 temp: float,
                 xrange: tuple,
                 yrange: tuple,
                 contourvals: list,
                 mesh: int = 200,
                 cmap: str = 'jet'):
        self.potential2D = potential2D
        self.kT = 8.3145 / 1000 * temp
        self.xrange = xrange
        self.yrange = yrange
        self.contourvals = contourvals
        self.mesh = mesh
        self.cmap = cmap

    def plot_potential(self):
        """
        Plots the potential within (xrange[0], xrange[1]) and (yrange[0], yrange[1])
        """
        grid_width = max(self.xrange[1] - self.xrange[0], self.yrange[1] - self.yrange[0]) / mesh
        xx, yy = np.mgrid[self.xrange[0]:self.xrange[1]:grid_width, self.yrange[0]:self.yrange[1]:grid_width]
        x = xx.ravel()
        y = yy.ravel()
        v = self.potential2D.potential(x, y)
        V = v.reshape(self.mesh, self.mesh) / self.kT

        fig, ax = plt.subplots(dpi=150)
        cs = ax.contourf(xx, yy, V, np.array(self.contourvals), cmap=self.cmap)
        cbar = fig.colorbar(cs)
        cbar.set_label(r"Free energy ($k_B T$)")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        return (fig, ax)

    def scatter_traj(self, traj, outimg):
        fig, ax = self.plot_potential()
        ax.scatter(traj[:, 0], traj[:, 1], s=1, c='black')
        plt.savefig(outimg)
        plt.close()

    def animate_traj(self, traj, outdir, call_ffmpeg: bool = True):
        for t in range(traj.shape[0]):
