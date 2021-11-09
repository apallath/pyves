import os

import matplotlib.pyplot as plt

from ves.bias import HarmonicBias_SingleParticle_x
from ves.config_creation import singleParticle2D_init_coord
from ves.langevin_dynamics import SingleParticleSimulation
from ves.visualization import VisualizePotential2D
from ves.potentials import SzaboBerezhkovskiiPotential as SBPotential

if not os.path.exists("harmonic_szabo_berezhkovskii_files/"):
    os.makedirs("harmonic_szabo_berezhkovskii_files/")

# Create and visualize potential energy surface
pot = SBPotential()
temp = 300
vis = VisualizePotential2D(pot, temp=temp,
                           xrange=[-7.5, 7.5], yrange=[-7.5, 7.5],
                           contourvals=[-2, -1, 0, 1, 2, 5, 8, 10])
fig, ax = vis.plot_potential()
plt.savefig("harmonic_szabo_berezhkovskii_files/potential.png")

# Monte carlo trials to place particle on potential energy surface
init_coord = singleParticle2D_init_coord(pot, 300, xmin=-7.5, xmax=7.5,
                                         ymin=-7.5, ymax=7.5)

# Perform single particle simulation
sim = SingleParticleSimulation(pot, init_coord=init_coord)

# Add a static harmonic bias potential along the x-coordinate to the
# landscape
harmonicbias = HarmonicBias_SingleParticle_x(k=200, x0=0, model_loc="harmonic_szabo_berezhkovskii_files/harmonic_bias_model.pt")
sim.init_ves(harmonicbias, static=True)

sim(nsteps=10000,
    chkevery=2000,
    trajevery=1,
    energyevery=1,
    chkfile="harmonic_szabo_berezhkovskii_files/chk_state.dat",
    trajfile="harmonic_szabo_berezhkovskii_files/traj.dat",
    energyfile="harmonic_szabo_berezhkovskii_files/energies.dat")

# Visualize trajectory
vis.scatter_traj(sim.traj, "harmonic_szabo_berezhkovskii_files/traj.png")
vis.animate_traj(sim.traj, "harmonic_szabo_berezhkovskii_files/traj_movie", every=50)
