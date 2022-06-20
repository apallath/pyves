"""
Classes for visualizing trajectory data.
"""
import os
from re import A

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
import torch
from tqdm import tqdm

from ves.potentials import Potential1D
from ves.potentials import Potential2D

# Resource-light and SSH-friendly non-GUI plotting
matplotlib.use('Agg')


class VisualizePotential1D:
    """
    Class defining functions to generate scatter plots and animated trajectories
    of a particle on a 1D potential.

    Args:
        potential1D:
        temp:
        xrange:
        mesh:
    """
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
        ax.set_ylabel(r"Potential ($k_B T$)")
        ax.set_xlabel("$x$")
        ax.set_ylim([0, None])
        return (fig, ax)

    def scatter_traj(self, traj, outimg, every=1, s=1, c='black'):
        """
        Scatters entire trajectory onto potential energy surface.

        Args:
            traj:
            outimg:
            every:
            s:
            c:
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

        Args:
            traj (iterable):
            outdir (str):
            every (int):
            s (int):
            c (str):
            call_ffmpeg (bool):
        """
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for t, frame in enumerate(tqdm(traj[::every])):
            fig, ax = self.plot_potential()
            xpt = frame[0]
            ypt = self.potential1D.potential(xpt) / self.kT
            ax.scatter(xpt, ypt, s=s, c=c)
            plt.savefig("{}/traj.{:05d}.png".format(outdir, t))
            plt.close()

        if call_ffmpeg:
            os.system("ffmpeg -r 25 -i {}/traj.%5d.png -vb 20M {}/traj.mp4".format(outdir, outdir))


class VisualizePotential2D:
    """
    Class defining functions to generate scatter plots and animated trajectories
    of a particle on a 2D potential surface. 
    
    The effect of a neural network bias on a landscape can be plotted
    over by passing a torch.nn.Module object through the `bias` argument and setting `biased = True`
    when calling any of the plotting functions.

    Args:
        potential2D (pyib.md.potentials.Potential2D): 2D potential energy surface.
        temp (float): Temperature (required, as free energies are plotted in kT).
        xrange (tuple of length 2): Range of x-values to plot.
        yrange (tuple of length 2): Range of y-values to plot.
        contourvals (int or array-like): Determines the number and positions of the contour lines / regions. Refer to the `matplotlib documentation`_ for details.
        clip (float): Value of free energy (in kT) to clip contour plot at.
        mesh: Number of mesh points in each dimension for contour plot.
        cmap: Matplotlib colormap.
        bias (torch.nn.Module): Bias potential.
        
    .. _matplotlib documentation: https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.pyplot.contour.html
    """
    def __init__(self,
                 potential2D: Potential2D,
                 temp: float,
                 xrange: tuple,
                 yrange: tuple,
                 contourvals=None,
                 clip=None,
                 mesh: int = 200,
                 cmap: str = 'jet',
                 bias: torch.nn.Module = None):
        self.potential2D = potential2D
        self.kT = 8.3145 / 1000 * temp
        self.xrange = xrange
        self.yrange = yrange
        self.contourvals = contourvals
        self.clip = clip
        self.mesh = mesh
        self.cmap = cmap
        self.bias = bias

    def plot_potential(self, biased=False):
        """
        Plots the potential within (xrange[0], xrange[1]) and (yrange[0], yrange[1]).
        """
        xx, yy = np.meshgrid(np.linspace(self.xrange[0], self.xrange[1], self.mesh), np.linspace(self.yrange[0], self.yrange[1], self.mesh))
        x = xx.ravel()
        y = yy.ravel()
        v = self.potential2D.potential(x, y)

        # Apply bias
        if biased and self.bias is not None:
            X_tensor = torch.tensor(np.vstack([x, y]).T).type(torch.DoubleTensor)
            b = self.bias(X_tensor).detach().numpy()
            v = v + b
            v -= v.min()

        if self.clip is not None:
            V = v.reshape(self.mesh, self.mesh) / self.kT
            V = V.clip(max=self.clip)
        else:
            V = v.reshape(self.mesh, self.mesh) / self.kT

        fig, ax = plt.subplots(dpi=150)
        if self.contourvals is not None:
            cs = ax.contourf(xx, yy, V, self.contourvals, cmap=self.cmap)
        else:
            cs = ax.contourf(xx, yy, V, cmap=self.cmap)
        cbar = fig.colorbar(cs)
        cbar.set_label(r"Free energy ($k_B T$)")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        return (fig, ax)

    def plot_projection_x(self, biased=False):
        """
        Plots the x-projection of potential within (xrange[0], xrange[1])
        and (yrange[0], yrange[1]).
        """
        # Compute 2D free energy profile
        xx, yy = np.meshgrid(np.linspace(self.xrange[0], self.xrange[1], self.mesh), np.linspace(self.yrange[0], self.yrange[1], self.mesh))
        x = xx.ravel()
        y = yy.ravel()
        v = self.potential2D.potential(x, y)

        # Apply bias
        if biased and self.bias is not None:
            X_tensor = torch.tensor(np.vstack([x, y]).T).type(torch.DoubleTensor)
            b = self.bias(X_tensor).detach().numpy()
            v = v + b
            v -= v.min()

        V = v.reshape(self.mesh, self.mesh) / self.kT

        # Integrate over y-coordinate to get free-energy along x-coordinate
        Fx = -logsumexp(-V, axis=0)
        Fx = Fx - np.min(Fx)
        x = np.linspace(self.xrange[0], self.xrange[1], self.mesh)

        # Plot
        fig, ax = plt.subplots(dpi=150)
        ax.plot(x, Fx)
        ax.set_ylabel(r"Potential ($k_B T$)")
        ax.set_xlabel("$x$")
        ax.set_ylim([0, self.clip])
        return (fig, ax, x, Fx)

    def plot_projection_y(self, biased=False):
        """
        Plots the y-projection of potential within (xrange[0], xrange[1])
        and (yrange[0], yrange[1]).
        """
        # Compute 2D free energy profile
        xx, yy = np.meshgrid(np.linspace(self.xrange[0], self.xrange[1], self.mesh), np.linspace(self.yrange[0], self.yrange[1], self.mesh))
        x = xx.ravel()
        y = yy.ravel()
        v = self.potential2D.potential(x, y)

        # Apply bias
        if biased and self.bias is not None:
            X_tensor = torch.tensor(np.vstack([x, y]).T).type(torch.DoubleTensor)
            b = self.bias(X_tensor).detach().numpy()
            v = v + b
            v -= v.min()

        V = v.reshape(self.mesh, self.mesh) / self.kT

        # Integrate over x-coordinate to get free-energy along x-coordinate
        Fy = -logsumexp(-V, axis=1)
        Fy = Fy - np.min(Fy)
        y = np.linspace(self.yrange[0], self.yrange[1], self.mesh)

        # Plot
        fig, ax = plt.subplots(dpi=150)
        ax.plot(y, Fy)
        ax.set_ylabel(r"Potential ($k_B T$)")
        ax.set_xlabel("$y$")
        ax.set_ylim([0, self.clip])
        return (fig, ax, y, Fy)

    def scatter_traj(self, traj, outimg, every=1, s=1, c='black', **plotkwargs):
        """
        Scatters entire trajectory onto potential energy surface.

        Args:
            traj:
            outimg:
            every:
            s:
            c:
        """
        fig, ax = self.plot_potential(**plotkwargs)
        ax.scatter(traj[::every, 0], traj[::every, 1], s=s, c=c)
        plt.savefig(outimg)
        plt.close()

    def scatter_traj_projection_x(self, traj, outimg, every=1, s=1, c='black', **plotkwargs):
        """
        Scatters x-projection of entire trajectory onto potential energy surface.

        Args:
            traj:
            outimg:
            every:
            s:
            c:
        """
        fig, ax, x, Fx = self.plot_projection_x(**plotkwargs)
        for i in tqdm(range(0, traj.shape[0], every)):
            xpt = traj[i, 0]
            yloc = np.argmin((x - xpt)**2)
            ypt = Fx[yloc]
            ax.scatter(xpt, ypt, s=s, c=c)
        plt.savefig(outimg)
        plt.close()

    def scatter_traj_projection_y(self, traj, outimg, every=1, s=1, c='black', **plotkwargs):
        """
        Scatters y-projection of entire trajectory onto potential energy surface.

        Args:
            traj:
            outimg:
            every:
            s:
            c:
        """
        fig, ax, y, Fy = self.plot_projection_y(**plotkwargs)
        for i in tqdm(range(0, traj.shape[0], every)):
            xpt = traj[i, 1]
            yloc = np.argmin((y - xpt)**2)
            ypt = Fy[yloc]
            ax.scatter(xpt, ypt, s=s, c=c)
        plt.savefig(outimg)
        plt.close()

    def animate_traj(self, traj, outdir, every=1, s=3, c='black', call_ffmpeg: bool = True, **plotkwargs):
        """
        Plots positions at timesteps defined by interval `every` on potential
        energy surface and stitches together plots using ffmpeg to make a movie.

        Args:
            traj (iterable):
            outdir (str):
            every (int):
            s (int):
            c (str):
            call_ffmpeg (bool):
        """
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for t, frame in enumerate(tqdm(traj[::every])):
            fig, ax = self.plot_potential(**plotkwargs)
            ax.scatter(frame[0], frame[1], s=s, c=c)
            plt.savefig("{}/traj.{:05d}.png".format(outdir, t))
            plt.close()

        if call_ffmpeg:
            os.system("ffmpeg -r 25 -i {}/traj.%5d.png -vb 20M {}/traj.mp4".format(outdir, outdir))

    def animate_traj_projection_x(self, traj, outdir, every=1, s=3, c='black',
                                  call_ffmpeg: bool = True, **plotkwargs):
        """
        Plots positions at timesteps defined by interval `every` on the x-projection of the
        potential energy surface and stitches together plots using ffmpeg to make a movie.

        Args:
            traj (iterable):
            outdir (str):
            every (int):
            s (int):
            c (str):
            call_ffmpeg (bool):
        """
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for t, frame in enumerate(tqdm(traj[::every])):
            fig, ax, x, Fx = self.plot_projection_x(**plotkwargs)
            xpt = frame[0]
            yloc = np.argmin((x - xpt)**2)
            ypt = Fx[yloc]
            ax.scatter(xpt, ypt, s=s, c=c)
            plt.savefig("{}/traj_x.{:05d}.png".format(outdir, t))
            plt.close()

        if call_ffmpeg:
            os.system("ffmpeg -r 25 -i {}/traj_x.%5d.png -vb 20M {}/traj_x.mp4".format(outdir, outdir))

    def animate_traj_projection_y(self, traj, outdir, every=1, s=3, c='black',
                                  call_ffmpeg: bool = True, **plotkwargs):
        """
        Plots positions at timesteps defined by interval `every` on the x-projection of the
        potential energy surface and stitches together plots using ffmpeg to make a movie.

        Args:
            traj (iterable):
            outdir (str):
            every (int):
            s (int):
            c (str):
            call_ffmpeg (bool):
        """
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for t, frame in enumerate(tqdm(traj[::every])):
            fig, ax, y, Fy = self.plot_projection_y(**plotkwargs)
            xpt = frame[1]
            yloc = np.argmin((y - xpt)**2)
            ypt = Fy[yloc]
            ax.scatter(xpt, ypt, s=s, c=c)
            plt.savefig("{}/traj_y.{:05d}.png".format(outdir, t))
            plt.close()

        if call_ffmpeg:
            os.system("ffmpeg -r 25 -i {}/traj_y.%5d.png -vb 20M {}/traj_y.mp4".format(outdir, outdir))


def visualize_path_CV_2D(xrange, yrange, mesh, x_i, y_i, lam, contourvals=None, cmap='jet', dpi=150):
    r"""
    Plots the parallel (s) and perpendicular (z) path CVs for the path defined by the 
    images (x_i, y_i) in the region defined by xrange and yrange.

    $$s = \frac{1}{N} \frac{\sum_{i=0}^{N-1} (i + 1)\ e^{-\lambda [(x - x_i) ^ 2 + (y - y_i) ^ 2]}}{\sum_{i=0}^{N-1} e^{-\lambda [(x - x_i) ^ 2 + (y - y_i) ^ 2]}}$$

    $$z = -\frac{1}{\lambda} \ln (\sum_{i=0}^{N-1} e^{-\lambda [(x - x_i) ^ 2 + (y - y_i) ^ 2]})$$

    Args:
        xrange (tuple of length 2): Range of x-values to plot.
        yrange (tuple of length 2): Range of y-values to plot.
        mesh: Number of mesh points in each dimension for contour plot.
        x_i: x-coordinates of images defining a path.
        y_i: y-coordinates of images defining a path.
        lam: Value of $\lambda$ for constructing path CVs.
        contourvals (int or array-like): Determines the number and positions of the contour lines / regions. Refer to the `matplotlib documentation`_ for details.
        cmap: Matplotlib colormap (default=jet).
        dpi: Output DPI (default=150).

    Returns:
        ((fig_s, ax_s), (fig_z, ax_z)): Figures and axes for each plot

    .. _matplotlib documentation: https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.pyplot.contour.html
    """
    xx, yy = np.meshgrid(np.linspace(xrange[0], xrange[1], mesh), np.linspace(yrange[0], yrange[1], mesh))
    x = xx.ravel()
    y = yy.ravel()

    assert(len(x_i) == len(y_i))
    ivals = np.arange(1, len(x_i) + 1)
    Npath = len(ivals)

    # Compute s
    s = 1 / Npath * np.exp(logsumexp(-lam * ((x[np.newaxis, :] - x_i[:, np.newaxis]) ** 2 + (y[np.newaxis, :] - y_i[:, np.newaxis]) ** 2) + np.log(ivals)[:, np.newaxis], axis=0) 
                           - logsumexp(-lam * ((x[np.newaxis, :] - x_i[:, np.newaxis]) ** 2 + (y[np.newaxis, :] - y_i[:, np.newaxis]) ** 2), axis=0))

    # Compute z
    z = -1 / lam * logsumexp(-lam * ((x[np.newaxis, :] - x_i[:, np.newaxis]) ** 2 + (y[np.newaxis, :] - y_i[:, np.newaxis]) ** 2), axis=0)

    # Plot s
    fig_s, ax_s = plt.subplots(dpi=dpi)
    if contourvals is not None:
        cs_s = ax_s.contourf(xx, yy, s.reshape(mesh, mesh).clip(min=0, max=1), contourvals, cmap=cmap)
    else:
        cs_s = ax_s.contourf(xx, yy, s.reshape(mesh, mesh).clip(min=0, max=1), cmap=cmap)
    cbar_s = fig_s.colorbar(cs_s)
    cbar_s.set_label(r"s")
    ax_s.set_xlabel("$x$")
    ax_s.set_ylabel("$y$")

    # Plot z
    fig_z, ax_z = plt.subplots(dpi=dpi)
    if contourvals is not None:
        cs_z = ax_z.contourf(xx, yy, z.reshape(mesh, mesh), contourvals, cmap=cmap)
    else:
        cs_z = ax_z.contourf(xx, yy, z.reshape(mesh, mesh), cmap=cmap)
    cbar_z = fig_z.colorbar(cs_z)
    cbar_z.set_label(r"z")
    ax_z.set_xlabel("$x$")
    ax_z.set_ylabel("$y$")

    return ((fig_s, ax_s), (fig_z, ax_z))


def visualize_free_energy_2D(xvals, yvals, xrange, yrange, nbins_x=100, nbins_y=100, contourvals=None, clip=None, cmap='jet', dpi=150):
    """
    Plots 2D free energy profile from 2D trajectory data.

    Args:
        xvals (numpy.ndarray): Array of x coordinates of points to bin.
        yvals (numpy.ndarray): Array of y coordinates of points to bin.
        xrange (tuple of length 2): Range of x-values to plot.
        yrange (tuple of length 2): Range of y-values to plot.
        nbins_x (int): Number of bins along the x-axis (default=100).
        nbins_y (int): Number of bins along the y-axis (default=100).
        contourvals (int or array-like): Determines the number and positions of the contour lines / regions. Refer to the `matplotlib documentation`_ for details.
        clip (float): Value of free energy (in kT) to clip contour plot at.
        cmap: Matplotlib colormap (default=jet).
        dpi: Output DPI (default=150).

    .. _matplotlib documentation: https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.pyplot.contour.html
    """
    # Compute betaF
    counts, xbins, ybins, img = plt.hist2d(xvals, yvals, range=[xrange, yrange], bins=[nbins_x, nbins_y])
    counts[counts == 0] = counts[counts != 0].min()
    betaF = -np.log(counts)
    betaF = betaF - np.min(betaF)

    # Plot contour vals
    fig, ax = plt.subplots(dpi=dpi)
    if contourvals is not None:
        cs = ax.contourf(betaF.T, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], levels=contourvals, cmap=cmap)
    else:
        cs = ax.contourf(betaF.T, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], cmap=cmap)
    cbar = fig.colorbar(cs)
    cbar.set_label(r"Free energy ($k_B T$)")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    return fig, ax
