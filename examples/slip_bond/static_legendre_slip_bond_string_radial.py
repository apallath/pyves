import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Important: This example requires apallath/stringmethod to compute the minimum energy path
# string and the profile along the minimum energy path
# Install this from github.com/apallath/stringmethod
import stringmethod

from ves.basis import LegendreBasis2DRadialCV
from ves.bias import StaticBias_SingleParticle
from ves.config_creation import singleParticle2D_init_coord
from ves.langevin_dynamics import SingleParticleSimulation
from ves.visualization import VisualizePotential2D
from ves.potentials import SlipBondPotential2D
from ves.utils import TrajectoryReader


if not os.path.exists("static_legendre_slip_bond_string_radial_files/"):
    os.makedirs("static_legendre_slip_bond_string_radial_files/")

# Create and visualize potential energy surface
pot = SlipBondPotential2D()
temp = 300

################################################################################
# Begin:  Fit legendre basis set expansion to string
################################################################################
fit_nn = True

if fit_nn:
    print("Computing string.")
    # Compute string
    x = np.linspace(-8, 10, 100)
    y = np.linspace(-6, 8, 100)
    xx, yy = np.meshgrid(x, y)
    V = 1000 / (8.314 * 300) * pot.potential(xx, yy)

    S = stringmethod.String2D(x, y, V)
    S.compute_mep(begin=[-4, -4], end=[5, 6], maxsteps=200, traj_every=10)

    # Plot
    fig, ax, cbar = S.plot_string_evolution(clip_max=20, levels=21, cmap='jet')
    fig.savefig("static_legendre_slip_bond_string_radial_files/string_evolution.png")
    fig, ax = S.plot_mep_energy_profile()
    fig.savefig("static_legendre_slip_bond_string_radial_files/string_energy_profile.png")

    # Get string coordinates and corresponding energies
    mep, Fmep = S.get_mep_energy_profile()

    # Reparameterize mep
    x_min = mep[0, 0]
    y_min = mep[0, 1]
    x_max = mep[-1, 0]
    y_max = mep[-1, 1]

    # Get s for each point
    s = np.zeros_like(Fmep)
    for ptidx in range(len(mep)):
        pt = mep[ptidx]
        s[ptidx] = ((pt[0] - x_min) ** 2 + (pt[1] - y_min) ** 2) / ((x_max - x_min) ** 2 + (y_max - y_min) ** 2)

    s = (s - 0.5) / 0.5

    fig, ax = plt.subplots(dpi=300)
    ax.plot(s, Fmep)
    fig.savefig("static_legendre_slip_bond_string_radial_files/string_energy_profile_reparam_s.png")

    print("Fitting string using legendre model.")

    # V(s) = -F(s), not -beta F(s)
    # Therefore, need to correct for effect of beta
    kT = 8.3145 / 1000 * temp
    Fmep = kT * Fmep

    mep = torch.tensor(mep)
    Fmep = torch.tensor(Fmep)
    nn_fn = LegendreBasis2DRadialCV(8, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, weights=None)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(nn_fn.parameters(), lr=1e-1)

    losses = []

    nnfitsteps = 10000
    lossoutevery = 2000

    for step in tqdm(range(nnfitsteps)):
        optimizer.zero_grad()
        output = nn_fn(mep)
        loss = loss_fn(output, Fmep)
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
    ax.plot(s, Fmep, label="Orig")
    fit_np = nn_fn(mep).detach().numpy()
    ax.plot(s, fit_np, label="Legendre fit")
    ax.set_xlabel("s")
    ax.set_ylabel("V(s)")
    ax.legend()
    plt.savefig("static_legendre_slip_bond_string_radial_files/potential_s_legendrefit.png")
    plt.close()

    fig, ax = plt.subplots(dpi=300)
    ax.plot(range(len(losses)), losses)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE loss")
    plt.savefig("static_legendre_slip_bond_string_radial_files/legendrefit_loss_history.png")
    plt.close()

    # Extract weights
    weights = nn_fn.weights.detach().numpy()
    # Negate
    weights = -weights

    # Print
    print(x_min, y_min, x_max, y_max)
    print(weights)

    # Save weights
    np.save("static_legendre_slip_bond_string_radial_files/minmax.npy", np.array([x_min, y_min, x_max, y_max]))
    np.save("static_legendre_slip_bond_string_radial_files/weights.npy", weights)

else:
    minmax = np.load("static_legendre_slip_bond_string_radial_files/minmax.npy")
    x_min = minmax[0]
    y_min = minmax[1]
    x_max = minmax[2]
    y_max = minmax[3]
    weights = np.load("static_legendre_slip_bond_string_radial_files/weights.npy")

################################################################################
# End: Fit legendre basis set expansion to string
################################################################################

################################################################################
# Begin: Plot biased landscape
################################################################################

vis = VisualizePotential2D(pot, temp=temp,
                           xrange=[-8, 10], yrange=[-6, 8],
                           contourvals=21, 
                           bias=LegendreBasis2DRadialCV(8, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, weights=weights),
                           clip=20)

# 2D surface
fig, ax = vis.plot_potential()
fig.savefig("static_legendre_slip_bond_string_radial_files/potential.png")
fig, ax = vis.plot_potential(biased=True)
fig.savefig("static_legendre_slip_bond_string_radial_files/potential_biased.png")

# 1D projection along x
fig, ax, _, _ = vis.plot_projection_x()
fig.savefig("static_legendre_slip_bond_string_radial_files/potential_x.png")
fig, ax, _, _ = vis.plot_projection_x(biased=True)
fig.savefig("static_legendre_slip_bond_string_radial_files/potential_x_biased.png")

fig, ax, _, _ = vis.plot_projection_y()
fig.savefig("static_legendre_slip_bond_string_radial_files/potential_y.png")
fig, ax, _, _ = vis.plot_projection_y(biased=True)
fig.savefig("static_legendre_slip_bond_string_radial_files/potential_y_biased.png")

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
    vis.scatter_traj(init_coord, "static_legendre_slip_bond_string_radial_files/init_coord.png", biased=True, c='white', s=2)

    # Perform single particle simulation
    sim = SingleParticleSimulation(pot, temp=temp, init_coord=init_coord, cpu_threads=1, traj_in_mem=False)

    # Begin: Initialize static bias
    V_module = LegendreBasis2DRadialCV(8, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, weights=weights)
    ves_bias = StaticBias_SingleParticle(V_module, model_loc="static_legendre_slip_bond_string_radial_files/model.pt")
    sim.init_ves(ves_bias, static=True, startafter=500)

    # Call simulation
    sim(nsteps=10 * 100000,  # run x ns simulation
        chkevery=5 * 10000,
        trajevery=1,
        energyevery=1,
        chkfile="static_legendre_slip_bond_string_radial_files/chk_state.dat",
        trajfile="static_legendre_slip_bond_string_radial_files/traj.dat",
        energyfile="static_legendre_slip_bond_string_radial_files/energies.dat")

################################################################################
# End: Simulation
################################################################################

################################################################################
# Begin: Plot timeseries & trajectories
################################################################################
t, traj = TrajectoryReader("static_legendre_slip_bond_string_radial_files/traj.dat").read_traj()

# Uncomment the following lines to plot trajectores:
vis.scatter_traj(traj, "static_legendre_slip_bond_string_radial_files/traj.png", c='white', every=50, biased=True)
vis.scatter_traj_projection_x(traj, "static_legendre_slip_bond_string_radial_files/traj_x.png", every=50, biased=True)
vis.scatter_traj_projection_y(traj, "static_legendre_slip_bond_string_radial_files/traj_y.png", every=50, biased=True)

# Uncomment the following lines to animate trajectores:
#vis.animate_traj(traj, "static_legendre_slip_bond_string_radial_files/traj_movie", c='white', s=2, every=200, biased=True)
#vis.animate_traj_projection_x(traj, "static_legendre_slip_bond_string_radial_files/traj_movie", every=200, biased=True)
#vis.animate_traj_projection_y(traj, "static_legendre_slip_bond_string_radial_files/traj_movie", every=200, biased=True)

fig, ax = plt.subplots(dpi=300)
ax.plot(t, traj[:, 0])
ax.set_ylim([-10, 12])
ax.set_xlabel("t")
ax.set_ylabel("x")
fig.savefig("static_legendre_slip_bond_string_radial_files/ts_x.png")

fig, ax = plt.subplots(dpi=300)
ax.plot(t, traj[:, 1])
ax.set_ylim([-8, 10])
ax.set_xlabel("t")
ax.set_ylabel("y")
fig.savefig("static_legendre_slip_bond_string_radial_files/ts_y.png")
