"""
Classes defining VES bias potentials and update mechanism
"""
from openmmtorch import TorchForce
import torch
import torch.optim as optim
import torch.nn as nn


class Bias:
    """
    Abstract class defining methods for VES bias to implement.
    """
    def __init__(self, model_loc="model.pt"):
        self.model_loc = model_loc

    def update(self, traj):
        pass

    @property
    def force(self):
        torch_force = TorchForce(self.model_loc)
        return torch_force

###############################################################################
# Simple harmonic potential bias
# E.g.: umbrella sampling along the x-coordinate
# No updates to the bias
###############################################################################


class HarmonicBias_SingleParticle_x(Bias):
    """
    Apply a harmonic bias potential along the x coordinate of a single particle.
    """
    def __init__(self, k, x0, model_loc="model.pt"):
        super().__init__(model_loc)
        module = torch.jit.script(HarmonicBias_SingleParticle_x_ForceModule(k, x0))
        module.save(self.model_loc)


class HarmonicBias_SingleParticle_x_ForceModule(torch.nn.Module):
    """
    A harmonic potential k/2 (x-x0)^2 as a static compute graph.

    The potential is only applied to the x-coordinate of the particle.
    """
    def __init__(self, k, x0):
        self.k = k
        self.x0 = x0
        super().__init__()

    def forward(self, positions):
        """The forward method returns the energy computed from positions.

        Args:
            positions : torch.Tensor with shape (nparticles, 3)
                positions[i,k] is the position (in nanometers) of spatial dimension k of particle i

        Returns:
            potential : torch.Scalar
                The potential energy (in kJ/mol)
        """
        return self.k / 2 * torch.sum((positions[:, 0] - self.x0) ** 2)


###############################################################################
# VES bias
# Applied potential V can either be:
# a) an expansion over a basis set (Valsson and Parrinello 2014)
# b) a neural network (Valsson and Parrinello 2014)
# This needs to be passed as an input parameter
# The update() setup updates the coefficients
###############################################################################


class VESBias_SingleParticle_x(Bias):
    """
    Apply a basis set expanded potential along the x coordinate of a single particle.
    """
    def __init__(self, V_module, target, beta, optimizer_type, optimizer_params, model_loc):
        # Bias potential V(x)
        self.model = V_module

        # Target distribution p(x)
        self.target = target

        # Required for computing loss function
        self.beta = beta

        if optimizer_type == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), **optimizer_params)
        elif optimizer_type == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), **optimizer_params)
        elif optimizer_type == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), **optimizer_params)
        else:
            raise ValueError("Requested optimizer not yet supported.")

    def update(self, traj):
        ########################################################################
        # Training loop with one optimizer step
        ########################################################################

        # Zero gradients
        self.optimizer.zero_grad()

        # Evaluate biased part of loss
        # Accumulate loss
        eBV = torch.tensor([0.0], requires_grad=True)
        for t in range(traj.shape[0]):
            x = torch.tensor(traj[t])
            eBV += torch.exp(self.beta * self.model(x))
        loss_V = 1 / self.beta * torch.log(eBV)

        # Evaluate target part of loss
        # Accumulate loss
        loss_p = torch.tensor([0.0], requires_grad=True)
        # TODO: accumulate loss

        # Sum loss
        loss = loss_V + loss_p

        # Track loss
        print(loss)

        # Backprop to compute gradients
        loss.backward()

        # Make an optimizer step
        self.optimizer.step()

        # Save updated model as torchscript
        module = torch.jit.script(self.model)
        module.save(self.model_loc)


################################################################################
# Bias potential zoo
# Add new bias potential forms here
################################################################################


class VESBias_SingleParticle_x_ForceModule_NN(torch.nn.Module):
    """
    Neural network bias with [16, 32, 64] architecture.
    """
    def __init__(self, basis_set_layer):
        super().__init__()

        self.fc1 = nn.linear(1, 16)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.linear(16, 32)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.linear(32, 64)
        self.tanh3 = nn.Tanh()
        self.fc4 = nn.linear(64, 1)

    def forward(self, positions):
        """The forward method returns the energy computed from positions.

        Args:
            positions : torch.Tensor with shape (nparticles, 3)
                positions[i,k] is the position (in nanometers) of spatial dimension k of particle i

        Returns:
            potential : torch.Scalar
                The potential energy (in kJ/mol)
        """
        # Extract x-coordinate
        x = positions[:, 0]

        # Apply NN bias
        bias = self.fc1(x)
        bias = self.tanh1(bias)
        bias = self.fc2(bias)
        bias = self.tanh2(bias)
        bias = self.fc3(bias)
        bias = self.tanh3(bias)
        bias = self.fc4(bias)

        # Return bias
        return bias
