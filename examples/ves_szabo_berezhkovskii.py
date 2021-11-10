import os

import matplotlib.pyplot as plt
import torch

from ves.bias import HarmonicBias_SingleParticle_x, HarmonicBias_SingleParticle_x_ForceModule
from ves.config_creation import singleParticle2D_init_coord
from ves.langevin_dynamics import SingleParticleSimulation
from ves.visualization import VisualizePotential2D
from ves.potentials import SzaboBerezhkovskiiPotential as SBPotential

if not os.path.exists("moving_harmonic_szabo_berezhkovskii_files/"):
    os.makedirs("moving_harmonic_szabo_berezhkovskii_files/")

# Create and visualize potential energy surface
pot = SBPotential()
temp = 300
vis = VisualizePotential2D(pot, temp=temp,
                           xrange=[-7.5, 7.5], yrange=[-7.5, 7.5],
                           contourvals=[-2, -1, 0, 1, 2, 5, 8, 10])

# 2D surface
fig, ax = vis.plot_potential()
plt.savefig("moving_harmonic_szabo_berezhkovskii_files/potential.png")
# 1D projection
fig, ax, _, _ = vis.plot_projection_x()
plt.savefig("moving_harmonic_szabo_berezhkovskii_files/potential_x.png")

# Monte carlo trials to place particle on potential energy surface
init_coord = singleParticle2D_init_coord(pot, 300, xmin=-7.5, xmax=7.5,
                                         ymin=-7.5, ymax=7.5)

# Perform single particle simulation
sim = SingleParticleSimulation(pot, init_coord=init_coord)


# Modify static harmonic bias to a dynamic harmonic bias
# The bias moves from -6 to 6 with increasing timestep
class DynamicHarmonicBias(HarmonicBias_SingleParticle_x):
    def update(self, traj):
        k = 200
        x0 = -6 + 12 * traj.shape[0] / 100000
        module = torch.jit.script(HarmonicBias_SingleParticle_x_ForceModule(k, x0))
        module.save(self.model_loc)


# Add dynamic harmonic bias potential along the x-coordinate to the
# landscape
harmonicbias = DynamicHarmonicBias(k=200, x0=-6,
                                   model_loc="moving_harmonic_szabo_berezhkovskii_files/harmonic_bias_model.pt")
sim.init_ves(harmonicbias, startafter=10000, learnevery=10000)

sim(nsteps=100000,
    chkevery=20000,
    trajevery=1,
    energyevery=1,
    chkfile="moving_harmonic_szabo_berezhkovskii_files/chk_state.dat",
    trajfile="moving_harmonic_szabo_berezhkovskii_files/traj.dat",
    energyfile="moving_harmonic_szabo_berezhkovskii_files/energies.dat")

# Visualize trajectory
vis.scatter_traj(sim.traj, "moving_harmonic_szabo_berezhkovskii_files/traj.png")
vis.scatter_traj_projection_x(sim.traj, "moving_harmonic_szabo_berezhkovskii_files/traj_x.png", every=50)
vis.animate_traj(sim.traj, "moving_harmonic_szabo_berezhkovskii_files/traj_movie", every=500)
vis.animate_traj_projection_x(sim.traj, "moving_harmonic_szabo_berezhkovskii_files/traj_movie", every=500)
