"""
Classes for visualizing trajectory data.
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from tqdm import tqdm

from ves.potentials import Potential1D
from ves.potentials import Potential2D

# Resource-light and SSH-friendly non-GUI plotting
matplotlib.use('Agg')


class VisualizePotential1D:
    def __init__(self,
                 potential1D: Potential1D,
                 temp: float,
                 xrange: tuple,
                 mesh: int = 200):
        self.potential1D = potential1D
        self.kT = 8.3145 / 1000 * temp
        self.xrange = xrange
        self.mesh = mesh

    def plot_potential(self):
        """
        Plots the potential within (xrange[0], xrange[1]).
        """
        x = np.linspace(self.xrange[0], self.xrange[1], self.mesh)
        V = self.potential1D.potential(x) / self.kT

        # Plot
        fig, ax = plt.subplots(dpi=150)
        ax.plot(x, V)
        ax.set_ylabel(r"Free energy ($k_B T$)")
        ax.set_xlabel("$x$")
        ax.set_ylim([0, None])
        return (fig, ax)

    def scatter_traj(self, traj, outimg, every=1, s=1, c='black'):
        """
        Scatters entire trajectory onto potential energy surface.
        """
        fig, ax = self.plot_potential()
        for i in tqdm(range(0, traj.shape[0], every)):
            xpt = traj[i, 0]
            ypt = self.potential1D.potential(xpt) / self.kT
            ax.scatter(xpt, ypt, s=s, c=c)
        plt.savefig(outimg)
        plt.close()

    def animate_traj(self, traj, outdir, every=1, s=3, c='black', call_ffmpeg: bool = True):
        """
        Plots positions at timesteps defined by interval `every` on potential
        energy surface and stitches together plots using ffmpeg to make a movie.
        """
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for t, frame in enumerate(tqdm(traj[::every])):
            fig, ax = self.plot_potential()
            ax.scatter(frame[0], frame[1], s=s, c=c)
            plt.savefig("{}/traj.{:05d}.png".format(outdir, t))
            plt.close()

        if call_ffmpeg:
            os.system("ffmpeg -r 25 -i {}/traj.%5d.png -vb 20M {}/traj.mp4".format(outdir, outdir))

    def animate_traj_projection_x(self, traj, outdir, every=1, s=3, c='black',
                                call_ffmpeg: bool = True):
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for t, frame in enumerate(tqdm(traj[::every])):
            fig, ax, x, Fx = self.plot_projection_x()
            xpt = frame[0]
            yloc = np.argmin((x - xpt)**2)
            ypt = Fx[yloc]
            ax.scatter(xpt, ypt, s=s, c=c)
            plt.savefig("{}/traj_x.{:05d}.png".format(outdir, t))
            plt.close()

        if call_ffmpeg:
            os.system("ffmpeg -r 25 -i {}/traj_x.%5d.png -vb 20M {}/traj_x.mp4".format(outdir, outdir))


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
        Plots the potential within (xrange[0], xrange[1]) and (yrange[0], yrange[1]).
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

    def plot_projection_x(self):
        """
        Plots the projection of potential within (xrange[0], xrange[1])
        and (yrange[0], yrange[1]) onto the x-axis.
        """
        # Compute 2D free energy profile
        grid_width = max(self.xrange[1] - self.xrange[0], self.yrange[1] - self.yrange[0]) / self.mesh
        xx, yy = np.mgrid[self.xrange[0]:self.xrange[1]:grid_width, self.yrange[0]:self.yrange[1]:grid_width]
        x = xx.ravel()
        y = yy.ravel()
        v = self.potential2D.potential(x, y)
        V = v.reshape(self.mesh, self.mesh) / self.kT

        # Integrate over y-coordinate to get free-energy along x-coordinate
        Fx = -logsumexp(-V, axis=1)
        Fx = Fx - np.min(Fx)
        x = np.arange(self.xrange[0], self.xrange[1], grid_width)

        # Plot
        fig, ax = plt.subplots(dpi=150)
        ax.plot(x, Fx)
        ax.set_ylabel(r"Free energy ($k_B T$)")
        ax.set_xlabel("$x$")
        ax.set_ylim([0, None])
        return (fig, ax, x, Fx)

    def scatter_traj(self, traj, outimg, every=1, s=1, c='black'):
        """
        Scatters entire trajectory onto potential energy surface.
        """
        fig, ax = self.plot_potential()
        ax.scatter(traj[::every, 0], traj[::every, 1], s=s, c=c)
        plt.savefig(outimg)
        plt.close()

    def scatter_traj_projection_x(self, traj, outimg, every=1, s=1, c='black'):
        """
        Scatters x-projection of entire trajectory onto potential energy surface.
        """
        fig, ax, x, Fx = self.plot_projection_x()
        for i in tqdm(range(0, traj.shape[0], every)):
            xpt = traj[i, 0]
            yloc = np.argmin((x - xpt)**2)
            ypt = Fx[yloc]
            ax.scatter(xpt, ypt, s=s, c=c)
        plt.savefig(outimg)
        plt.close()

    def animate_traj(self, traj, outdir, every=1, s=3, c='black', call_ffmpeg: bool = True):
        """
        Plots positions at timesteps defined by interval `every` on potential
        energy surface and stitches together plots using ffmpeg to make a movie.
        """
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for t, frame in enumerate(tqdm(traj[::every])):
            fig, ax = self.plot_potential()
            ax.scatter(frame[0], frame[1], s=s, c=c)
            plt.savefig("{}/traj.{:05d}.png".format(outdir, t))
            plt.close()

        if call_ffmpeg:
            os.system("ffmpeg -r 25 -i {}/traj.%5d.png -vb 20M {}/traj.mp4".format(outdir, outdir))

    def animate_traj_projection_x(self, traj, outdir, every=1, s=3, c='black',
                                call_ffmpeg: bool = True):
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for t, frame in enumerate(tqdm(traj[::every])):
            fig, ax, x, Fx = self.plot_projection_x()
            xpt = frame[0]
            yloc = np.argmin((x - xpt)**2)
            ypt = Fx[yloc]
            ax.scatter(xpt, ypt, s=s, c=c)
            plt.savefig("{}/traj_x.{:05d}.png".format(outdir, t))
            plt.close()

        if call_ffmpeg:
            os.system("ffmpeg -r 25 -i {}/traj_x.%5d.png -vb 20M {}/traj_x.mp4".format(outdir, outdir))
