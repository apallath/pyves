import os

import matplotlib.pyplot as plt

from ves.config_creation import singleParticle1D_init_coord
from ves.langevin_dynamics import SingleParticleSimulation
from ves.visualization import VisualizePotential1D
from ves.potentials import SingleWellPotential1D

if not os.path.exists("unbiased_single_well_files/"):
    os.makedirs("unbiased_single_well_files/")

# Create and a single_well potential energy surface
pot = SingleWellPotential1D()
temp = 300
vis = VisualizePotential1D(pot, temp=temp, xrange=[-7.5, 7.5])

# Plot potential
fig, ax = vis.plot_potential()
plt.savefig("unbiased_single_well_files/potential.png")

# Monte carlo trials to place particle on potential energy surface
init_coord = singleParticle1D_init_coord(pot, 300, xmin=-7.5, xmax=7.5)

# Perform single particle simulation
sim = SingleParticleSimulation(pot, init_coord=init_coord)

sim(nsteps=10000,
    chkevery=2000,
    trajevery=1,
    energyevery=1,
    chkfile="unbiased_single_well_files/chk_state.dat",
    trajfile="unbiased_single_well_files/traj.dat",
    energyfile="unbiased_single_well_files/energies.dat")

# Visualize trajectory
vis.scatter_traj(sim.traj, "unbiased_single_well_files/traj.png")
vis.animate_traj(sim.traj, "unbiased_single_well_files/traj_movie", every=50)
