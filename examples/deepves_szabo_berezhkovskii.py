import os

import matplotlib.pyplot as plt
import torch

from ves.bias import VESBias_SingleParticle_x, VESBias_SingleParticle_x_ForceModule_NN
from ves.config_creation import singleParticle2D_init_coord
from ves.langevin_dynamics import SingleParticleSimulation
from ves.potentials import SzaboBerezhkovskiiPotential as SBPotential
from ves.target import Target_Uniform_HardSwitch_x
from ves.visualization import VisualizePotential2D


if not os.path.exists("deepves_szabo_berezhkovskii_files/"):
    os.makedirs("deepves_szabo_berezhkovskii_files/")

# Create and visualize potential energy surface
pot = SBPotential()
temp = 300
vis = VisualizePotential2D(pot, temp=temp,
                           xrange=[-7.5, 7.5], yrange=[-7.5, 7.5],
                           contourvals=[-2, -1, 0, 1, 2, 5, 8, 10])

# 2D surface
fig, ax = vis.plot_potential()
plt.savefig("deepves_szabo_berezhkovskii_files/potential.png")
# 1D projection
fig, ax, _, _ = vis.plot_projection_x()
plt.savefig("deepves_szabo_berezhkovskii_files/potential_x.png")

################################################################################
# Begin: Fit neural network to 1D projection
################################################################################

################################################################################
# End: Fit neural network to 1D projection
################################################################################

# Monte carlo trials to place particle on potential energy surface
init_coord = singleParticle2D_init_coord(pot, 300, xmin=-7.5, xmax=7.5,
                                         ymin=-7.5, ymax=7.5)

# Initialize single particle simulation
sim = SingleParticleSimulation(pot, init_coord=init_coord)

################################################################################
# Begin: Initialize VES bias
################################################################################
beta = 1 / (8.3145 / 1000 * temp)
target = Target_Uniform_HardSwitch_x(-7.5, 7.5, 200)
V_module = VESBias_SingleParticle_x_ForceModule_NN()

ves_bias = VESBias_SingleParticle_x(V_module,
                                    target,
                                    beta,
                                    optimizer_type="Adam",
                                    optimizer_params={'lr': 0.01},
                                    model_loc="deepves_szabo_berezhkovskii_files/harmonic_bias_model.pt")

sim.init_ves(ves_bias, startafter=500, learnevery=500)

################################################################################
# End: Initialize VES bias
################################################################################

sim(nsteps=10000,
    chkevery=2000,
    trajevery=1,
    energyevery=1,
    chkfile="deepves_szabo_berezhkovskii_files/chk_state.dat",
    trajfile="deepves_szabo_berezhkovskii_files/traj.dat",
    energyfile="deepves_szabo_berezhkovskii_files/energies.dat")

# Visualize trajectory
vis.scatter_traj(sim.traj, "deepves_szabo_berezhkovskii_files/traj.png")
vis.scatter_traj_projection_x(sim.traj, "deepves_szabo_berezhkovskii_files/traj_x.png", every=50)
vis.animate_traj(sim.traj, "deepves_szabo_berezhkovskii_files/traj_movie", every=500)
vis.animate_traj_projection_x(sim.traj, "deepves_szabo_berezhkovskii_files/traj_movie", every=500)
