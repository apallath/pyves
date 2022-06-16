import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Important: This example requires apallath/stringmethod to compute the minimum energy path
# string and the profile along the minimum energy path
# Install this from github.com/apallath/stringmethod
import stringmethod

from ves.basis import LegendreBasisString2D
from ves.bias import StaticBias_SingleParticle
from ves.config_creation import singleParticle2D_init_coord
from ves.langevin_dynamics import SingleParticleSimulation
from ves.visualization import VisualizePotential2D
from ves.potentials import DoubleWellPotential2D
from ves.utils import TrajectoryReader


if not os.path.exists("static_legendre_double_well_string_files/"):
    os.makedirs("static_legendre_double_well_string_files/")

# Create and visualize potential energy surface
pot = DoubleWellPotential2D()
temp = 300
vis = VisualizePotential2D(pot, temp=temp,
                           xrange=[-2.25, 2.25], yrange=[-2, 2],
                           contourvals=11)

# 2D surface
fig, ax = vis.plot_potential()
fig.savefig("static_legendre_double_well_string_files/potential.png")

################################################################################
# Begin:  Fit legendre basis set expansion to string
################################################################################
fit_nn = True
if fit_nn:
    print("Computing string.")
    # Compute string
    x = np.linspace(-2.25, 2.25, 100)
    y = np.linspace(-2.25, 2.25, 100)
    xx, yy = np.meshgrid(x, y)
    V = 1000 / (8.314 * 300) * pot.potential(xx, yy)

    S = stringmethod.String2D(x, y, V)
    S.compute_mep(begin=[-1.5, 0], end=[1.5, 0], maxsteps=200, traj_every=10)

    # Plot
    fig, ax, cbar = S.plot_string_evolution(levels=61, cmap='jet')
    fig.savefig("static_legendre_double_well_string_files/string_evolution.png")
    fig, ax = S.plot_mep_energy_profile()
    fig.savefig("static_legendre_double_well_string_files/string_energy_profile.png")

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
    fig.savefig("static_legendre_double_well_string_files/string_energy_profile_reparam_s.png")

    print("Fitting string using legendre model.")

    # V(s) = -F(s), not -beta F(s)
    # Therefore, need to correct for effect of beta
    kT = 8.3145 / 1000 * temp
    Fmep = kT * Fmep

    mep = torch.tensor(mep)
    Fmep = torch.tensor(Fmep)
    nn_fn = LegendreBasisString2D(8, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, weights=None)

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
    plt.savefig("static_legendre_double_well_string_files/potential_s_legendrefit.png")
    plt.close()

    fig, ax = plt.subplots(dpi=300)
    ax.plot(range(len(losses)), losses)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE loss")
    plt.savefig("static_legendre_double_well_string_files/legendrefit_loss_history.png")
    plt.close()

    # Extract weights
    weights = nn_fn.weights.detach().numpy()
    # Negate
    weights = -weights

    # Print
    print(x_min, y_min, x_max, y_max)
    print(weights)

################################################################################
# End: Fit legendre basis set expansion to string
################################################################################

################################################################################
# Begin: Simulation
################################################################################
run_sim = True
if run_sim:
    # Monte carlo trials to place particle on potential energy surface
    init_coord = singleParticle2D_init_coord(pot, temp, xmin=-2.25, xmax=2.25,
                                             ymin=-2, ymax=2)

    # Perform single particle simulation
    sim = SingleParticleSimulation(pot, temp=temp, init_coord=init_coord, cpu_threads=1)

    # Begin: Initialize static bias
    if fit_nn is False:
        x_min = None
        y_min = None
        x_max = None
        y_max = None
        weights = np.array([])
    V_module = LegendreBasisString2D(8, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, weights=weights)
    ves_bias = StaticBias_SingleParticle(V_module, model_loc="static_legendre_double_well_string_files/model.pt")
    sim.init_ves(ves_bias, static=True, startafter=500)

    # Call simulation
    sim(nsteps=200000,  # run 2ns simulation
        chkevery=10000,
        trajevery=1,
        energyevery=1,
        chkfile="static_legendre_double_well_string_files/chk_state.dat",
        trajfile="static_legendre_double_well_string_files/traj.dat",
        energyfile="static_legendre_double_well_string_files/energies.dat")

    # Visualize trajectory
    vis.scatter_traj(sim.traj, "static_legendre_double_well_string_files/traj.png", every=50)
    vis.scatter_traj_projection_x(sim.traj, "static_legendre_double_well_string_files/traj_x.png", every=50)
    vis.scatter_traj_projection_y(sim.traj, "static_legendre_double_well_string_files/traj_y.png", every=50)

    # Uncomment the following lines for animated trajectores:
    #vis.animate_traj(sim.traj, "static_legendre_double_well_string_files/traj_movie", every=200)
    #vis.animate_traj_projection_x(sim.traj, "static_legendre_double_well_string_files/traj_movie", every=200)
    #vis.animate_traj_projection_y(sim.traj, "static_legendre_double_well_string_files/traj_movie", every=200)

################################################################################
# End: Simulation
################################################################################

################################################################################
# Begin: Plot timeseries & trajectories
################################################################################
t, traj = TrajectoryReader("static_legendre_double_well_string_files/traj.dat").read_traj()

fig, ax = plt.subplots(dpi=300)
ax.plot(t, traj[:, 0])
ax.set_xlabel("t")
ax.set_ylabel("x")
fig.savefig("static_legendre_double_well_string_files/ts_x.png")

# Uncomment the following lines to re-plot trajectores:
#vis.scatter_traj(traj, "static_legendre_double_well_string_files/traj.png", every=50)
#vis.scatter_traj_projection_x(traj, "static_legendre_double_well_string_files/traj_x.png", every=50)
#vis.scatter_traj_projection_y(traj, "static_legendre_double_well_string_files/traj_y.png", every=50)

# Uncomment the following lines to re-animate trajectores:
#vis.animate_traj(traj, "static_legendre_double_well_string_files/traj_movie", every=200)
#vis.animate_traj_projection_x(traj, "static_legendre_double_well_string_files/traj_movie", every=200)
#vis.animate_traj_projection_y(traj, "static_legendre_double_well_string_files/traj_movie", every=200)
