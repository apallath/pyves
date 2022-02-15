"""
Classes defining various VES basis expansions
"""
import torch
import torch.nn as nn


torch.set_default_tensor_type(torch.DoubleTensor)


class LegendreBasis(nn.Module):
    """
    Legendre polynomial basis with user-defined degree.
    """
    def __init__(self, degree):
        self.degree = degree
        self.weights = torch.nn.parameter(torch.randn(degree + 1))

    def legendre_polynomial(self, x, degree):
        pass

    def forward(self, positions):
        """The forward method returns the energy computed from positions.

        Args:
            positions : torch.Tensor with shape (1, 3)
                positions[0, k] is the position (in nanometers) of spatial dimension k of particle 0

        Returns:
            potential : torch.Scalar
                The potential energy (in kJ/mol)
        """
        # Extract x-coordinate
        x = positions[:, 0]

        # Apply legendre expansion bias
        bias = torch.zeros_like(x)
        for i in range(self.degree):
            bias += self.weights[i] * self.legendre_polynomial(x, i)

        return bias


class NNBasis(nn.Module):
    """
    Neural network basis with a user-defined architecture.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 48)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(48, 24)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(24, 1)

    def forward(self, positions):
        """The forward method returns the energy computed from positions.

        Args:
            positions : torch.Tensor with shape (1, 3)
                positions[0, k] is the position (in nanometers) of spatial dimension k of particle 0

        Returns:
            potential : torch.Scalar
                The potential energy (in kJ/mol)
        """
        # Extract x-coordinate
        x = positions[:, 0]

        # Apply NN bias
        bias = self.fc1(x)
        bias = self.act1(bias)
        bias = self.fc2(bias)
        bias = self.act2(bias)
        bias = self.fc3(bias)

        # Return bias
        return bias
