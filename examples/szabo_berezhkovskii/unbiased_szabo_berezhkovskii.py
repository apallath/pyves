import os

import matplotlib.pyplot as plt

from ves.config_creation import singleParticle2D_init_coord
from ves.langevin_dynamics import SingleParticleSimulation
from ves.visualization import VisualizePotential2D
from ves.potentials import SzaboBerezhkovskiiPotential as SBPotential

if not os.path.exists("unbiased_szabo_berezhkovskii_files/"):
    os.makedirs("unbiased_szabo_berezhkovskii_files/")

# Create and visualize potential energy surface
pot = SBPotential()
temp = 300
vis = VisualizePotential2D(pot, temp=temp,
                           xrange=[-7.5, 7.5], yrange=[-7.5, 7.5],
                           contourvals=[-2, -1, 0, 1, 2, 5, 8, 10])

# 2D surface
fig, ax = vis.plot_potential()
fig.savefig("unbiased_szabo_berezhkovskii_files/potential.png")

# 1D projections
fig, ax, _, _ = vis.plot_projection_x()
fig.savefig("unbiased_szabo_berezhkovskii_files/potential_x.png")
fig, ax, _, _ = vis.plot_projection_y()
fig.savefig("unbiased_szabo_berezhkovskii_files/potential_y.png")

plt.close('all')

# Monte carlo trials to place particle on potential energy surface
init_coord = singleParticle2D_init_coord(pot, 300, xmin=-7.5, xmax=7.5,
                                         ymin=-7.5, ymax=7.5)

# Perform single particle simulation
sim = SingleParticleSimulation(pot, init_coord=init_coord)


sim(nsteps=10000,
    chkevery=2000,
    trajevery=1,
    energyevery=1,
    chkfile="unbiased_szabo_berezhkovskii_files/chk_state.dat",
    trajfile="unbiased_szabo_berezhkovskii_files/traj.dat",
    energyfile="unbiased_szabo_berezhkovskii_files/energies.dat")

# Visualize trajectory
vis.scatter_traj(sim.traj, "unbiased_szabo_berezhkovskii_files/traj.png", every=50)
vis.scatter_traj_projection_x(sim.traj, "unbiased_szabo_berezhkovskii_files/traj_x.png", every=50)
vis.scatter_traj_projection_y(sim.traj, "unbiased_szabo_berezhkovskii_files/traj_y.png", every=50)
vis.animate_traj(sim.traj, "unbiased_szabo_berezhkovskii_files/traj_movie", every=200)
vis.animate_traj_projection_x(sim.traj, "unbiased_szabo_berezhkovskii_files/traj_movie", every=200)
vis.animate_traj_projection_y(sim.traj, "unbiased_szabo_berezhkovskii_files/traj_movie", every=200)
