import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from ves.basis import NNBasis1D
from ves.bias import VESBias_SingleParticle_x
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
                           xrange=[-4, 4], yrange=[-4, 4],
                           contourvals=[-2, -1, 0, 1, 2, 5, 8, 10])

# 2D surface
fig, ax = vis.plot_potential()
plt.savefig("deepves_szabo_berezhkovskii_files/potential.png")

# 1D projection
fig, ax, x, Fx = vis.plot_projection_x()
plt.savefig("deepves_szabo_berezhkovskii_files/potential_x.png")

fit_nn = False
if fit_nn:
    ################################################################################
    # Begin: Fit neural network to 1D projection
    # This is to test the fitting capacity of the model
    ################################################################################
    print("Fitting x-projection using NN model.")

    x_np = x[:]
    # Min/max
    min = x.min()
    max = x.max()

    x = torch.tensor(x).unsqueeze(-1).unsqueeze(-1)
    Fx = torch.tensor(Fx).unsqueeze(-1)
    nn_fn = NNBasis1D(min=min, max=max, hidden_layer_sizes=[128, 128])
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(nn_fn.parameters(), lr=3e-4)

    losses = []

    nnfitsteps = 10000
    lossoutevery = 1000

    for step in tqdm(range(nnfitsteps)):
        optimizer.zero_grad()
        output = nn_fn(x)
        loss = loss_fn(output, Fx)
        losses.append(loss.detach().numpy())
        loss.backward()
        optimizer.step()

        if step % lossoutevery == 0:
            print("{:.4e}".format(loss.detach().numpy()))

    # 1D projection fit
    fit_np = nn_fn(x).detach().numpy()
    ax.plot(x_np, fit_np, label="NN fit")
    ax.legend()
    plt.savefig("deepves_szabo_berezhkovskii_files/potential_x_nnfit.png")
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(range(nnfitsteps), losses)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE loss")
    plt.savefig("deepves_szabo_berezhkovskii_files/nnfit_loss_history.png")
    plt.close()
    ################################################################################
    # End: Fit neural network to 1D projection
    ################################################################################

run_sim = True
if run_sim:
    # Monte carlo trials to place particle on potential energy surface
    init_coord = singleParticle2D_init_coord(pot, 300, xmin=-4, xmax=4,
                                             ymin=-4, ymax=4)

    # Initialize single particle simulation
    sim = SingleParticleSimulation(pot, init_coord=init_coord)

    ################################################################################
    # Begin: Initialize VES bias
    ################################################################################
    beta = 1 / (8.3145 / 1000 * temp)
    target = Target_Uniform_HardSwitch_x(200)
    V_module = NNBasis1D(min=-4, max=4, hidden_layer_sizes=[128, 128])

    ves_bias = VESBias_SingleParticle_x(V_module,
                                        target,
                                        beta,
                                        optimizer_type="Adam",
                                        optimizer_params={'lr': 0.001},
                                        model_loc="deepves_szabo_berezhkovskii_files/bias_model.pt")

    sim.init_ves(ves_bias, startafter=500, learnevery=500)
    ################################################################################
    # End: Initialize VES bias
    ################################################################################

    sim(nsteps=100000,
        chkevery=20000,
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

################################################################################
# Begin: Plot VES bias evolution
# Loads saved models from file and plots them
################################################################################
paths = sorted(pathlib.Path('deepves_szabo_berezhkovskii_files').glob('bias_model.pt.iter*'))
iters = []
for path in paths:
    iters.append(int(str(path).split(".iter")[1]))
iters.sort()
print(iters)

if not os.path.exists('deepves_szabo_berezhkovskii_files/deepves_bias_movie'):
    os.makedirs('deepves_szabo_berezhkovskii_files/deepves_bias_movie')

print("Plotting bias evolution...")

for iteridx, iter in enumerate(tqdm(iters)):
    fig, ax, x, Fx = vis.plot_projection_x()

    # Standardize input
    mu_x = np.mean(x)
    sigma_x = np.std(x)
    x_scale = (x - mu_x) / sigma_x

    # Convert input to tensor
    x_scale = torch.tensor(x_scale).reshape((len(x_scale), 1, 1))

    # Load model
    model = torch.jit.load('deepves_szabo_berezhkovskii_files/bias_model.pt.iter{}'.format(iter))
    output = model(x_scale).detach().numpy()
    applied_bias = -output
    applied_bias = applied_bias - np.min(applied_bias)
    ax.plot(x, applied_bias)

    # Save to file
    plt.savefig('deepves_szabo_berezhkovskii_files/deepves_bias_movie/iter{}.png'.format(iter))
    plt.savefig('deepves_szabo_berezhkovskii_files/deepves_bias_movie/iter.{:05d}.png'.format(iteridx))

    # Close
    plt.close()

os.system("ffmpeg -r 25 -i deepves_szabo_berezhkovskii_files/deepves_bias_movie/iter.%05d.png -vb 20M deepves_szabo_berezhkovskii_files/deepves_bias_movie/movie.mp4")

################################################################################
# Begin: End VES bias evolution
################################################################################
