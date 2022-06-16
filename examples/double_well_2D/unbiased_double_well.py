import os

import matplotlib.pyplot as plt

from ves.config_creation import singleParticle2D_init_coord
from ves.langevin_dynamics import SingleParticleSimulation
from ves.visualization import VisualizePotential2D
from ves.potentials import DoubleWellPotential2D

if not os.path.exists("unbiased_double_well_files/"):
    os.makedirs("unbiased_double_well_files/")

# Create and visualize potential energy surface
pot = DoubleWellPotential2D()
temp = 300
vis = VisualizePotential2D(pot, temp=temp,
                           xrange=[-2.5, 2.5], yrange=[-2, 2],
                           contourvals=61)

# 2D surface
fig, ax = vis.plot_potential()
fig.savefig("unbiased_double_well_files/potential.png")

# 1D projections
fig, ax, _, _ = vis.plot_projection_x()
fig.savefig("unbiased_double_well_files/potential_x.png")
fig, ax, _, _ = vis.plot_projection_y()
fig.savefig("unbiased_double_well_files/potential_y.png")

plt.close('all')

# Monte carlo trials to place particle on potential energy surface
init_coord = singleParticle2D_init_coord(pot, temp, xmin=-2.5, xmax=2.5,
                                         ymin=-2, ymax=2)

# Perform single particle simulation
sim = SingleParticleSimulation(pot, temp=temp, init_coord=init_coord)


sim(nsteps=200000,
    chkevery=10000,
    trajevery=1,
    energyevery=1,
    chkfile="unbiased_double_well_files/chk_state.dat",
    trajfile="unbiased_double_well_files/traj.dat",
    energyfile="unbiased_double_well_files/energies.dat")

# Visualize trajectory
vis.scatter_traj(sim.traj, "unbiased_double_well_files/traj.png", every=50)
vis.scatter_traj_projection_x(sim.traj, "unbiased_double_well_files/traj_x.png", every=50)
vis.scatter_traj_projection_y(sim.traj, "unbiased_double_well_files/traj_y.png", every=50)

# Uncomment the following lines for animated trajectores:
#vis.animate_traj(sim.traj, "unbiased_double_well_files/traj_movie", every=200)
#vis.animate_traj_projection_x(sim.traj, "unbiased_double_well_files/traj_movie", every=200)
#vis.animate_traj_projection_y(sim.traj, "unbiased_double_well_files/traj_movie", every=200)
