import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Important: This example requires apallath/stringmethod to compute the minimum energy path
# string and the profile along the minimum energy path
# Install this from github.com/apallath/stringmethod
import stringmethod

from ves.basis import LegendreBasis1D
from ves.bias import StaticBias_SingleParticle
from ves.config_creation import singleParticle2D_init_coord
from ves.langevin_dynamics import SingleParticleSimulation
from ves.visualization import VisualizePotential2D
from ves.potentials import SlipBondPotential2D
from ves.utils import TrajectoryReader


if not os.path.exists("static_legendre_slip_bond_string_files/"):
    os.makedirs("static_legendre_slip_bond_string_files/")

# Create and visualize potential energy surface
pot = SlipBondPotential2D()
temp = 300
vis = VisualizePotential2D(pot, temp=temp,
                           xrange=[-8, 10], yrange=[-6, 8],
                           contourvals=61)

# 2D surface
fig, ax = vis.plot_potential()
fig.savefig("static_legendre_slip_bond_string_files/potential.png")

################################################################################
# Begin: String computation & fitting
################################################################################
# Compute string
x = np.linspace(-8, 10, 100)
y = np.linspace(-6, 8, 100)
xx, yy = np.meshgrid(x, y)
V = 1000 / (8.314 * 300) * pot.potential(xx, yy)

S = stringmethod.String2D(x, y, V)
S.compute_mep(begin=[-4, -4], end=[5, 6], maxsteps=200, traj_every=10)
fig, ax, cbar = S.plot_string_evolution(levels=61, cmap='jet')
fig.savefig("static_legendre_slip_bond_string_files/string_evolution.png")

################################################################################
# End: String computation & fitting
################################################################################

################################################################################
# Begin: Fit legendre basis set expansion to string
################################################################################
fit_nn = False
if fit_nn:
    print("Fitting y-projection using legendre model.")

    y_np = y[:]
    # V(s) = -F(s), not -beta F(s)
    # Therefore, need to correct for effect of beta
    kT = 8.3145 / 1000 * temp
    Fy = kT * Fy

    y = torch.tensor(y).unsqueeze(-1)
    Fy = torch.tensor(Fy)
    nn_fn = LegendreBasis1D(5, min=-6, max=8, axis='x', weights=None)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(nn_fn.parameters(), lr=1e-1)

    losses = []

    nnfitsteps = 10000
    lossoutevery = 2000

    for step in tqdm(range(nnfitsteps)):
        optimizer.zero_grad()
        output = nn_fn(y)
        loss = loss_fn(output, Fy)
        losses.append(loss.detach().numpy())
        loss.backward()
        optimizer.step()

        if loss < 1e-8:
            print("Converged in {} steps".format(step))
            break

        if step % lossoutevery == 0:
            print("{:.4e}".format(loss.detach().numpy()))

    # 1D projection fit
    fig, ax = plt.subplots(dpi=300)
    ax.plot(y, Fy, label="Orig")
    fit_np = nn_fn(y).detach().numpy()
    ax.plot(y_np, fit_np, label="Legendre fit")
    ax.set_xlabel("y")
    ax.set_ylabel("V(y)")
    ax.legend()
    plt.savefig("static_legendre_slip_bond_string_files/potential_y_legendrefit.png")
    plt.close()

    fig, ax = plt.subplots(dpi=300)
    ax.plot(range(len(losses)), losses)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE loss")
    plt.savefig("static_legendre_slip_bond_string_files/legendrefit_loss_history.png")
    plt.close()

    # Extract weights
    weights = nn_fn.weights.detach().numpy()
    # Negate
    weights = -weights
    # Print
    print(weights)

################################################################################
# End: Fit legendre basis set expansion to string
################################################################################

################################################################################
# Begin: Simulation
################################################################################
run_sim = False
if run_sim:
    # Monte carlo trials to place particle on potential energy surface
    init_coord = singleParticle2D_init_coord(pot, temp, xmin=-8, xmax=10,
                                             ymin=-6, ymax=8)

    # Perform single particle simulation
    sim = SingleParticleSimulation(pot, temp=temp, init_coord=init_coord, cpu_threads=1)

    # Begin: Initialize static bias
    if fit_nn is False:
        weights = np.array([-9.09672317e+00, -5.70681000e-07, -2.71365067e+00, -5.87584016e-07, -2.20418439e+01])
    V_module = LegendreBasis1D(5, min=-6, max=8, axis='y', weights=weights)
    ves_bias = StaticBias_SingleParticle(V_module, model_loc="static_legendre_slip_bond_string_files/model.pt")
    sim.init_ves(ves_bias, static=True, startafter=500)

    # Call simulation
    sim(nsteps=200000,  # run 2ns simulation
        chkevery=10000,
        trajevery=1,
        energyevery=1,
        chkfile="static_legendre_slip_bond_string_files/chk_state.dat",
        trajfile="static_legendre_slip_bond_string_files/traj.dat",
        energyfile="static_legendre_slip_bond_string_files/energies.dat")

    # Visualize trajectory
    vis.scatter_traj(sim.traj, "static_legendre_slip_bond_string_files/traj.png", every=50)
    vis.scatter_traj_projection_x(sim.traj, "static_legendre_slip_bond_string_files/traj_x.png", every=50)
    vis.scatter_traj_projection_y(sim.traj, "static_legendre_slip_bond_string_files/traj_y.png", every=50)

    # Uncomment the following lines for animated trajectores:
    #vis.animate_traj(sim.traj, "static_legendre_slip_bond_string_files/traj_movie", every=200)
    #vis.animate_traj_projection_x(sim.traj, "static_legendre_slip_bond_string_files/traj_movie", every=200)
    #vis.animate_traj_projection_y(sim.traj, "static_legendre_slip_bond_string_files/traj_movie", every=200)

################################################################################
# End: Simulation
################################################################################

################################################################################
# Begin: Plot timeseries & trajectories
################################################################################
t, traj = TrajectoryReader("static_legendre_slip_bond_string_files/traj.dat").read_traj()

fig, ax = plt.subplots(dpi=300)
ax.plot(t, traj[:, 0])
ax.set_xlabel("t")
ax.set_ylabel("y")
fig.savefig("static_legendre_slip_bond_string_files/ts_y.png")

# Uncomment the following lines to re-plot trajectores:
#vis.scatter_traj(traj, "static_legendre_slip_bond_string_files/traj.png", every=50)
#vis.scatter_traj_projection_x(traj, "static_legendre_slip_bond_string_files/traj_x.png", every=50)
#vis.scatter_traj_projection_y(traj, "static_legendre_slip_bond_string_files/traj_y.png", every=50)

# Uncomment the following lines to re-animate trajectores:
#vis.animate_traj(traj, "static_legendre_slip_bond_string_files/traj_movie", every=200)
#vis.animate_traj_projection_x(traj, "static_legendre_slip_bond_string_files/traj_movie", every=200)
#vis.animate_traj_projection_y(traj, "static_legendre_slip_bond_string_files/traj_movie", every=200)
