"""
Classes defining VES bias potentials and update mechanism
"""
from openmmtorch import TorchForce
import torch
import torch.optim


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
# VES (Valsson and Parrinello 2014)
# Applied potential V is an expansion over a basis set
# The update() setup updates the coefficients
###############################################################################


class BasisSetExpansionBias_SingleParticle_x(Bias):
    """
    Apply a basis set expanded potential along the x coordinate of a single particle.
    """
    def __init__(self, basis_set_layer, optimizer_type, optimizer_params, model_loc):
        self.model = BasisSetExpansionBias_SingleParticle_x_ForceModule(basis_set_layer)

        if optimizer_type == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), **optimizer_params)
        elif optimizer_type == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), **optimizer_params)
        elif optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_params)
        else:
            raise ValueError("Requested optimizer not yet supported.")

    def update(self, traj):
        ########################################################################
        # Training loop with one optimizer step
        ########################################################################

        # Zero gradients
        self.optimizer.zero_grad()

        # Evaluate biased part of loss
        # Accumulate gradients
        loss = None

        # Evaluate target part of loss
        # Accumulate gradients

        # Backprop to compute gradients
        loss.backward()

        # Make an optimizer step
        self.optimizer.step()

        # Save updated model as torchscript
        module = torch.jit.script(self.model)
        module.save(self.model_loc)


class BasisSetExpansionBias_SingleParticle_x_ForceModule(torch.nn.Module):
    """
    Basis set expansion bias.
    """
    def __init__(self, basis_set_layer):
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
        # Extract x-coordinate
        xpos = positions[:, 0]

        # Apply bias to x-coordinate
        bias = self.V(xpos)

        # Return bias
        return bias

###############################################################################
# VES (Valsson and Parrinello 2014)
# Applied potential V is a neural network
# The update() setup updates the NN parameters
###############################################################################


class NeuralNetworkBias_SingleParticle_x(Bias):
    """
    Apply a neural network bias potential along the x coordinate of a single particle.
    """
    def __init__(self, NN, model_loc):
        pass

    def update(self, traj):
        pass
