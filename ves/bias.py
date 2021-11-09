"""
Classes defining VES bias potentials
"""
from openmmtorch import TorchForce
import torch


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


class HarmonicBias_SingleParticle_x(Bias):
    """
    Apply a harmonic bias potential along the x coordinate of a single particle.
    """
    def __init__(self, k, x0, model_loc="model.pt"):
        super().__init__(model_loc)
        module = torch.jit.script(HarmonicBias_SingleParticle_x_ForceModule(k, x0))
        module.save(self.model_loc)

    def update(self, traj, k, x0):
        # Ignore traj variable
        # Set harmonic bias at new location
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


class BasisSetExpansionBias_SingleParticle_x(Bias):
    """
    Apply a basis set expanded potential along the x coordinate of a single particle.
    """
    def __init__(self, BasisSet, model_loc):
        pass

    def update(self, traj):
        pass


class NeuralNetworkBias_SingleParticle_x(Bias):
    """
    Apply a neural network bias potential along the x coordinate of a single particle.
    """
    def __init__(self, NN, model_loc):
        pass

    def update(self, traj):
        pass
