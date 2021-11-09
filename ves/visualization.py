"""
Classes for visualizing trajectory data.
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ves.potentials import Potential2D

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
        grid_width = max(self.xrange[1] - self.xrange[0], self.yrange[1] - self.yrange[0]) / self.mesh
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

    def scatter_traj(self, traj, outimg, every=1, s=1, c='black'):
        fig, ax = self.plot_potential()
        ax.scatter(traj[::every, 0], traj[::every, 1], s=s, c=c)
        plt.savefig(outimg)
        plt.close()

    def animate_traj(self, traj, outdir, every=1, s=3, c='black', call_ffmpeg: bool = True):
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for t, frame in enumerate(tqdm(traj[::every])):
            fig, ax = self.plot_potential()
            ax.scatter(frame[0], frame[1], s=s, c=c)
            plt.savefig("{}/traj.{:05d}.png".format(outdir, t))
            plt.close()

        if call_ffmpeg:
            os.system("ffmpeg -r 25 -i {}/traj.%5d.png -vb 20M {}/traj.mp4".format(outdir, outdir))
