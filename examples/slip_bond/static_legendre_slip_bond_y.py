import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from ves.basis import LegendreBasis1D
from ves.bias import StaticBias_SingleParticle
from ves.config_creation import singleParticle2D_init_coord
from ves.langevin_dynamics import SingleParticleSimulation
from ves.visualization import VisualizePotential2D
from ves.potentials import SlipBondPotential2D
from ves.utils import TrajectoryReader


if not os.path.exists("static_legendre_slip_bond_y_files/"):
    os.makedirs("static_legendre_slip_bond_y_files/")

# Create and visualize potential energy surface
pot = SlipBondPotential2D()
temp = 300
vis = VisualizePotential2D(pot, temp=temp,
                           xrange=[-8, 10], yrange=[-4, 6],
                           contourvals=61)

# 1D projection along y
_, _, y, Fy = vis.plot_projection_y()

################################################################################
# Begin: Fit legendre basis set expansion to 1D projection
################################################################################
fit_nn = True
if fit_nn:
    print("Fitting y-projection using legendre model.")

    y_np = y[:]
    # V(s) = -F(s), not -beta F(s)
    # Therefore, need to correct for effect of beta
    kT = 8.3145 / 1000 * temp
    Fy = kT * Fy

    y = torch.tensor(y).unsqueeze(-1)
    Fy = torch.tensor(Fy)
    nn_fn = LegendreBasis1D(5, min=-4, max=6, axis='x', weights=None)

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
    plt.savefig("static_legendre_slip_bond_y_files/potential_y_legendrefit.png")
    plt.close()

    fig, ax = plt.subplots(dpi=300)
    ax.plot(range(len(losses)), losses)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE loss")
    plt.savefig("static_legendre_slip_bond_y_files/legendrefit_loss_history.png")
    plt.close()

    # Extract weights
    weights = nn_fn.weights.detach().numpy()
    # Negate
    weights = -weights
    # Print
    print(weights)

    # Save weights
    np.save("static_legendre_slip_bond_y_files/weights.npy", weights)

else:
    weights = np.load("static_legendre_slip_bond_y_files/weights.npy")

################################################################################
# End: Fit legendre basis set expansion to 1D projection
################################################################################

################################################################################
# Begin: Plot biased landscape
################################################################################

vis = VisualizePotential2D(pot, temp=temp,
                           xrange=[-8, 10], yrange=[-6, 8],
                           contourvals=21, 
                           bias=LegendreBasis1D(5, min=-4, max=6, axis='y', weights=weights),
                           clip=20)

# 2D surface
fig, ax = vis.plot_potential()
fig.savefig("static_legendre_slip_bond_y_files/potential.png")
fig, ax = vis.plot_potential(biased=True)
fig.savefig("static_legendre_slip_bond_y_files/potential_biased.png")

# 1D projection along x
fig, ax, _, _ = vis.plot_projection_x()
fig.savefig("static_legendre_slip_bond_y_files/potential_x.png")
fig, ax, _, _ = vis.plot_projection_x(biased=True)
fig.savefig("static_legendre_slip_bond_y_files/potential_x_biased.png")

fig, ax, _, _ = vis.plot_projection_y()
fig.savefig("static_legendre_slip_bond_y_files/potential_y.png")
fig, ax, _, _ = vis.plot_projection_y(biased=True)
fig.savefig("static_legendre_slip_bond_y_files/potential_y_biased.png")

################################################################################
# End: Plot biased landscape
################################################################################

################################################################################
# Begin: Simulation
################################################################################
run_sim = True
if run_sim:
    # Monte carlo trials to place particle on potential energy surface
    init_coord = singleParticle2D_init_coord(pot, temp, xmin=-8, xmax=10,
                                             ymin=-6, ymax=8)

    # Plot initial coordinate
    vis.scatter_traj(init_coord, "static_legendre_slip_bond_y_files/init_coord.png", biased=True, c='white', s=2)

    # Perform single particle simulation
    sim = SingleParticleSimulation(pot, temp=temp, init_coord=init_coord, cpu_threads=1, traj_in_mem=False)

    # Begin: Initialize static bias
    V_module = LegendreBasis1D(5, min=-4, max=6, axis='y', weights=weights)
    ves_bias = StaticBias_SingleParticle(V_module, model_loc="static_legendre_slip_bond_y_files/model.pt")
    sim.init_ves(ves_bias, static=True, startafter=500)

    # Call simulation
    sim(nsteps=10 * 100000,  # run x ns simulation
        chkevery=10000,
        trajevery=1,
        energyevery=1,
        chkfile="static_legendre_slip_bond_y_files/chk_state.dat",
        trajfile="static_legendre_slip_bond_y_files/traj.dat",
        energyfile="static_legendre_slip_bond_y_files/energies.dat")

################################################################################
# End: Simulation
################################################################################

################################################################################
# Begin: Plot traj and timeseries
################################################################################
t, traj = TrajectoryReader("static_legendre_slip_bond_y_files/traj.dat").read_traj()

# Uncomment the following lines to plot trajectores:
vis.scatter_traj(traj, "static_legendre_slip_bond_y_files/traj.png",  c='white', every=50, biased=True)
vis.scatter_traj_projection_x(traj, "static_legendre_slip_bond_y_files/traj_x.png",  every=50, biased=True)
vis.scatter_traj_projection_y(traj, "static_legendre_slip_bond_y_files/traj_y.png",  every=50, biased=True)

# Uncomment the following lines to animate trajectores:
#vis.animate_traj(traj, "static_legendre_slip_bond_y_files/traj_movie", c='white', s=2, every=200, biased=True)
#vis.animate_traj_projection_x(traj, "static_legendre_slip_bond_y_files/traj_movie", every=200, biased=True)
#vis.animate_traj_projection_y(traj, "static_legendre_slip_bond_y_files/traj_movie", every=200, biased=True)

fig, ax = plt.subplots(dpi=300)
ax.plot(t, traj[:, 0])
ax.set_ylim([-10, 12])
ax.set_xlabel("t")
ax.set_ylabel("x")
fig.savefig("static_legendre_slip_bond_y_files/ts_x.png")

fig, ax = plt.subplots(dpi=300)
ax.plot(t, traj[:, 1])
ax.set_ylim([-8, 10])
ax.set_xlabel("t")
ax.set_ylabel("y")
fig.savefig("static_legendre_slip_bond_y_files/ts_y.png")


