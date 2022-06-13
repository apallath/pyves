"""
Classes defining VES basis expansions for use with the neural network VES bias.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.set_default_tensor_type(torch.DoubleTensor)


class LegendreBasis_x(nn.Module):
    """
    Legendre polynomial basis with user-defined degree.

    Note: the size of the computational graph required to compute gradients
    increases with the degree of the legendre polynomial.
    """
    def __init__(self, degree):
        self.degree = degree
        self.weights = nn.parameter.Parameter(torch.randn(degree + 1))

    @classmethod
    def legendre_polynomial(cls, x, degree):
        r"""
        Computes a legendre polynomial of degree $n$ using dynamic programming
        and the Bonnet's recursion formula:
        $$(n + 1) P_{n+1}(x) = (2n + 1) x P_n(x) - nP_{n-1}(x)$$
        """
        if degree == 0:
            return torch.ones(x.size(0), requires_grad=True).type(x.type())

        elif degree == 1:
            return x

        elif degree > 1:
            P_n_minus = torch.ones(x.size(0), requires_grad=True).type(x.type())
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
        # Extract x-coordinate
        x = positions[:, 0]

        # Apply legendre expansion bias
        bias = torch.zeros_like(x)
        for i in range(self.degree):
            bias += self.weights[i] * self.legendre_polynomial(x, i)

        return bias


class NNBasis_x(nn.Module):
    """
    Neural network basis with a user-defined architecture.
    """
    def __init__(self, input_size=1, output_size=1, hidden_layer_sizes=[]):
        super().__init__()
        self.linears = nn.ModuleList()

        if len(hidden_layer_sizes) == 0:
            raise ValueError("NNBasis needs at least one hidden layer")

        # Input layer
        self.linears.append(nn.Linear(input_size, hidden_layer_sizes[0]))

        # Middle layers
        for lidx in range(1, len(hidden_layer_sizes)):
            self.linears.append(nn.Linear(hidden_layer_sizes[lidx - 1], hidden_layer_sizes[lidx]))

        # Output layer
        self.linears.append(nn.Linear(hidden_layer_sizes[-1], output_size))

    def forward(self, positions):
        """The forward method returns the energy computed from positions.

        Args:
            positions : torch.Tensor with shape (1, 3)
                positions[0, k] is the position (in nanometers) of spatial dimension k of particle 0

        Returns:
            potential : torch.Scalar
                The potential energy (in kJ/mol)
        """
        x = positions[:, 0]  # Extract x-coordinate

        for linear in self.linears[:-1]:
            x = F.relu(linear(x))
        x = self.linears[-1](x)

        return x
