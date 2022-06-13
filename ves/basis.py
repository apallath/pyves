"""
Classes defining VES basis expansions for use with the neural network VES bias.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.set_default_tensor_type(torch.DoubleTensor)


class LegendreBasis1D(torch.nn.Module):
    """
    Legendre polynomial basis expansion along x- or y- coordinate,
    with user-defined degree.

    Attributes:
        degree (int)
        axis (str): 'x' or 'y' (default='x')
    """
    def __init__(self, degree, weights, min, max, axis='x'):
        super().__init__()
        self.degree = degree
        self.weights = torch.from_numpy(weights).type(torch.DoubleTensor)
        self.min = min
        self.max = max
        self.axis = axis

    @classmethod
    def legendre_polynomial(cls, x, degree: int) -> torch.Tensor:
        r"""
        Computes a legendre polynomial of degree $n$ using dynamic programming
        and the Bonnet's recursion formula:

        $$(n + 1) P_{n+1}(x) = (2n + 1) x P_n(x) - nP_{n-1}(x)$$
        """
        if degree == 0:
            ones_list = x.size(0) * [1.0]
            return torch.tensor(ones_list, requires_grad=True).type(x.type())

        elif degree == 1:
            return x

        else:
            ones_list = x.size(0) * [1.0]
            P_n_minus = torch.tensor(ones_list, requires_grad=True).type(x.type())
            P_n = x

            for n in range(1, degree):
                P_n_plus = ((2 * n + 1) * x * P_n - n * P_n_minus) / (n + 1)

                # Replace
                P_n_minus = P_n
                P_n = P_n_plus
            return P_n

    def forward(self, positions):
        """The forward method returns the energy computed from positions.

        Args:
            positions : torch.Tensor with shape (1, 3)
                positions[0, k] is the position (in nanometers) of spatial dimension k of particle 0

        Returns:
            potential : torch.Scalar
                The potential energy (in kJ/mol)
        """
        # Extract coordinate
        if self.axis == 'x':
            x = positions[:, 0]
        elif self.axis == 'y':
            x = positions[:, 1]
        else:
            raise ValueError("Invalid axis")

        # Scale from [-1, 1]
        x = (x - (self.min + self.max) / 2) / ((self.max - self.min) / 2)

        # Apply legendre expansion bias
        bias = torch.zeros_like(x)
        for i in range(self.degree):
            bias += self.weights[i] * self.legendre_polynomial(x, i)
        return bias


class NNBasis1D(nn.Module):
    """
    Neural network basis with a user-defined architecture.
    """
    def __init__(self, min, max, axis='x', input_size=1, output_size=1, hidden_layer_sizes=[]):
        super().__init__()

        self.min = min
        self.max = max
        self.axis = axis

        self.layers = nn.ModuleList()

        if len(hidden_layer_sizes) == 0:
            raise ValueError("NNBasis needs at least one hidden layer")

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_layer_sizes[0]))

        # Middle layers
        for lidx in range(1, len(hidden_layer_sizes)):
            self.layers.append(nn.Linear(hidden_layer_sizes[lidx - 1], hidden_layer_sizes[lidx]))
            self.layers.append(nn.ReLU())

        # Output layer
        self.layers.append(nn.Linear(hidden_layer_sizes[-1], output_size))

    def forward(self, positions):
        """The forward method returns the energy computed from positions.

        Args:
            positions : torch.Tensor with shape (1, 3)
                positions[0, k] is the position (in nanometers) of spatial dimension k of particle 0

        Returns:
            potential : torch.Scalar
                The potential energy (in kJ/mol)
        """
        # Extract coordinate
        if self.axis == 'x':
            x = positions[:, 0]
        elif self.axis == 'y':
            x = positions[:, 1]
        else:
            raise ValueError("Invalid axis")

        # Scale from [0, 1]
        x = (x - self.min) / (self.max - self.min)

        for layer in self.layers:
            x = layer(x)

        return x
