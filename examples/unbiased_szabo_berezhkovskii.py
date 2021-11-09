import os

import matplotlib.pyplot as plt
import numpy as np
from ves.langevin_dynamics import SingleParticleSimulation
from ves.visualization import VisualizePotential2D
from ves.potentials import SzaboBerezhkovskiiPotential as SBPotential

# Create and visualize potential energy surface
pot = SBPotential()
temp = 300
vis = VisualizePotential2D(pot, temp=temp,
                           xrange=[-7.5, 7.5], yrange=[-7.5, 7.5],
                           contourvals=[-2, -1, 0, 1, 2, 5, 8, 10])
fig, ax = vis.plot_potential()

if not os.path.exists("unbiased_szabo_berezhkovskii_files/"):
    os.makedirs("unbiased_szabo_berezhkovskii_files/")

plt.savefig("unbiased_szabo_berezhkovskii_files/potential.png")

#Importance sampling to place particle on potential energy surface
init_coord = None
for i in range(100):
    trial_coord = [15 * np.random.random() - 7.5,
                   15 * np.random.random() - 7.5,
                   0]
    beta = 1 / (8.3145 / 1000 * temp)
    boltzmann_factor = np.exp(-beta * pot.potential(trial_coord[0], trial_coord[1]))
    if boltzmann_factor >= 0.5:
        init_coord = trial_coord
        print("Particle placement succeeded at iteration {}".format(i))
        break

if init_coord is None:
    raise RuntimeError("Could not place particle on surface.")

init_coord = np.array(init_coord).reshape((1, 3))

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
vis.scatter_traj(sim.traj, "unbiased_szabo_berezhkovskii_files/traj.png")
vis.animate_traj(sim.traj, "unbiased_szabo_berezhkovskii_files/traj_movie", every=50)
