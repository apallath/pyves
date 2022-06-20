import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Important: This example requires apallath/stringmethod to compute the minimum energy path
# string and the profile along the minimum energy path
# Install this from github.com/apallath/stringmethod
import stringmethod

from ves.basis import LegendreBasis2DPathCV
from ves.bias import StaticBias_SingleParticle
from ves.config_creation import singleParticle2D_init_coord
from ves.langevin_dynamics import SingleParticleSimulation
from ves.visualization import VisualizePotential2D, visualize_path_CV_2D
from ves.potentials import SlipBondPotential2D
from ves.utils import TrajectoryReader


if not os.path.exists("static_legendre_slip_bond_string_path_files/"):
    os.makedirs("static_legendre_slip_bond_string_path_files/")

# Create and visualize potential energy surface
pot = SlipBondPotential2D()
temp = 300
vis = VisualizePotential2D(pot, temp=temp,
                           xrange=[-8, 10], yrange=[-6, 8],
                           contourvals=21, clip=20)

# 2D surface
fig, ax = vis.plot_potential()
fig.savefig("static_legendre_slip_bond_string_path_files/potential.png")

################################################################################
# Begin: Compute string and plot path CVs
################################################################################
compute_string = False

if compute_string:
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
    fig.savefig("static_legendre_slip_bond_string_path_files/string_evolution.png")

    fig, ax = S.plot_mep_energy_profile()
    fig.savefig("static_legendre_slip_bond_string_path_files/string_energy_profile.png")

    # Get string coordinates and corresponding energies
    mep, Fmep = S.get_mep_energy_profile()

    # Plot Path CVs along mep
    ((fig_s, ax_s), (fig_z, ax_z)) = visualize_path_CV_2D(xrange=(-8, 10), yrange=(-8, 10), mesh=100, x_i=mep[:, 0], y_i=mep[:, 1], lam=500, contourvals=50, dpi=300, cmap="RdYlBu")
    ax_s.plot(mep[:, 0], mep[:, 1], color='white')
    fig_s.savefig("static_legendre_slip_bond_string_path_files/path_s.png")
    ax_z.plot(mep[:, 0], mep[:, 1], color='white')
    fig_z.savefig("static_legendre_slip_bond_string_path_files/path_z.png")

    # Save string
    np.save("static_legendre_slip_bond_string_path_files/mep.npy", mep)
    np.save("static_legendre_slip_bond_string_path_files/Fmep.npy", Fmep)

else:
    mep = np.load("static_legendre_slip_bond_string_path_files/mep.npy")
    F_mep = np.load("static_legendre_slip_bond_string_path_files/Fmep.npy")

x_i = mep[:, 0]
y_i = mep[:, 1]

################################################################################
# End: Compute string and plot path CVs
################################################################################

################################################################################
# Begin:  Fit legendre basis set expansion to string
################################################################################
fit_nn = False

if fit_nn:
    print("Fitting string using legendre model.")

    # V(s) = -F(s), not -beta F(s)
    # Therefore, need to correct for effect of beta
    kT = 8.3145 / 1000 * temp
    Fmep = kT * Fmep

    mep = torch.tensor(mep)
    Fmep = torch.tensor(Fmep)

    nn_fn = LegendreBasis2DPathCV(8, x_i=x_i, y_i=y_i, lam=100, weights=None)

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
    ax.plot(np.linspace(0, 1, len(mep)), Fmep, label="Orig")
    fit_np = nn_fn(mep).detach().numpy()
    ax.plot(np.linspace(0, 1, len(mep)), fit_np, label="Legendre fit")
    ax.set_xlabel("s")
    ax.set_ylabel("V(s)")
    ax.legend()
    plt.savefig("static_legendre_slip_bond_string_path_files/potential_s_legendrefit.png")
    plt.close()

    fig, ax = plt.subplots(dpi=300)
    ax.plot(range(len(losses)), losses)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE loss")
    plt.savefig("static_legendre_slip_bond_string_path_files/legendrefit_loss_history.png")
    plt.close()

    # Extract weights
    weights = nn_fn.weights.detach().numpy()
    # Negate
    weights = -weights

    # Print
    print(weights)
    
    # Save weights
    np.save("static_legendre_slip_bond_string_path_files/weights.npy", weights)

else:
    weights = np.load("static_legendre_slip_bond_string_path_files/weights.npy")

################################################################################
# End: Fit legendre basis set expansion to string
################################################################################

################################################################################
# Begin: Plot biased landscape
################################################################################

vis = VisualizePotential2D(pot, temp=temp,
                           xrange=[-8, 10], yrange=[-6, 8],
                           contourvals=21, 
                           bias=LegendreBasis2DPathCV(8, x_i=x_i, y_i=y_i, lam=100, weights=weights),
                           clip=20)

# 2D surface
fig, ax = vis.plot_potential(biased=True)
fig.savefig("static_legendre_slip_bond_string_path_files/biased_landscape.png")

################################################################################
# End: Plot biased landscape
################################################################################

################################################################################
# Begin: Simulation
################################################################################
run_sim = False

if run_sim:
    # Monte carlo trials to place particle on potential energy surface
    init_coord = singleParticle2D_init_coord(pot, temp, xmin=-8, xmax=10,
                                             ymin=-6, ymax=8)
    
    # Plot initial coordinate
    vis.scatter_traj(init_coord, "static_legendre_slip_bond_string_path_files/init_coord.png", biased=True, c='white', s=2)

    # Perform single particle simulation
    sim = SingleParticleSimulation(pot, temp=temp, init_coord=init_coord, cpu_threads=1)

    # Begin: Initialize static bias
    V_module = LegendreBasis2DPathCV(8, x_i=x_i, y_i=y_i, lam=100, weights=weights)
    ves_bias = StaticBias_SingleParticle(V_module, model_loc="static_legendre_slip_bond_string_path_files/model.pt")
    sim.init_ves(ves_bias, static=True, startafter=500)

    # Call simulation
    sim(nsteps=200000,  # run 2ns simulation
        chkevery=10000,
        trajevery=1,
        energyevery=1,
        chkfile="static_legendre_slip_bond_string_path_files/chk_state.dat",
        trajfile="static_legendre_slip_bond_string_path_files/traj.dat",
        energyfile="static_legendre_slip_bond_string_path_files/energies.dat")

################################################################################
# End: Simulation
################################################################################

################################################################################
# Begin: Plot traj and timeseries
################################################################################
t, traj = TrajectoryReader("static_legendre_slip_bond_string_path_files/traj.dat").read_traj()

# Uncomment the following lines to plot trajectores:
vis.scatter_traj(traj, "static_legendre_slip_bond_string_path_files/traj.png", c='white', every=50, biased=True)
vis.scatter_traj_projection_x(traj, "static_legendre_slip_bond_string_path_files/traj_x.png", c='white', every=50, biased=True)
vis.scatter_traj_projection_y(traj, "static_legendre_slip_bond_string_path_files/traj_y.png", c='white', every=50, biased=True)

# Uncomment the following lines to animate trajectores:
#vis.animate_traj(traj, "static_legendre_slip_bond_string_path_files/traj_movie", c='white', s=2, every=200, biased=True)
#vis.animate_traj_projection_x(traj, "static_legendre_slip_bond_string_path_files/traj_movie", c='white', s=2, every=200, biased=True)
#vis.animate_traj_projection_y(traj, "static_legendre_slip_bond_string_path_files/traj_movie", c='white', s=2, every=200, biased=True)

fig, ax = plt.subplots(dpi=300)
ax.plot(t, traj[:, 0])
ax.set_xlabel("t")
ax.set_ylabel("x")
fig.savefig("static_legendre_slip_bond_string_path_files/ts_x.png")

fig, ax = plt.subplots(dpi=300)
ax.plot(t, traj[:, 1])
ax.set_xlabel("t")
ax.set_ylabel("y")
fig.savefig("static_legendre_slip_bond_string_path_files/ts_y.png")


