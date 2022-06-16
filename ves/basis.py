"""
Classes defining basis expansions.
"""
import torch
import torch.nn as nn


torch.set_default_tensor_type(torch.DoubleTensor)

################################################################################
# 1D basis expansions: along x or y axes.
################################################################################


class LegendreBasis1D(torch.nn.Module):
    r"""
    Legendre polynomial basis expansion along x- or y- coordinate.

    The coordinates x (or y) are scaled to lie within [-1, 1] using the
    [min, max] attributes, as

    $$x' = (x - (min + max) / 2) / ((max - min) / 2)$$

    A legendre basis expansion is defined over x', as

    $$B(x') = \sum{i=0}^{d} w_i P_i(x')$$

    where P_i is the legendre polynomial of order i, w_i is its coefficient in
    the expansion, and d is the degree of the expansion.

    Notes:
        - Weights w_i are learnable as self.weights is a torch Parameter.
        - This basis expansion can directly be used a TorchForce module.

    Attributes:
        degree (int): Degree of expansion.
        min (float): Min x-/y-value for scaling.
        max (float): Max x-/y-value for scaling.
        axis (str): 'x' or 'y' (default='x').
        weights (torch.nn.Parameter): Legendre polynomial coefficients (array len = degree).

    Args:
        degree (int): Degree of expansion.
        min (float): Min x-/y-value for scaling.
        max (float): Max x-/y-value for scaling.
        axis (str): 'x' or 'y' (default='x').
        weights (numpy.ndarray): Legendre polynomial coefficients (array len = degree).
    """
    def __init__(self, degree, min, max, axis='x', weights=None):
        super().__init__()
        self.degree = degree
        self.min = min
        self.max = max
        self.axis = axis

        if weights is not None:
            weights_tensor = torch.from_numpy(weights).type(torch.DoubleTensor)
        else:
            weights_tensor = torch.rand(degree).type(torch.DoubleTensor)
        self.weights = nn.Parameter(weights_tensor)

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

        # Clamp to prevent x from going outside Legendre polynomial domain
        # Also restricts enhanced sampling to within [min, max]
        x = torch.clamp(x, min=-1, max=1)

        # Apply legendre expansion bias
        bias = torch.zeros_like(x)
        for i in range(self.degree):
            bias += self.weights[i] * self.legendre_polynomial(x, i)
        return bias


class NNBasis1D(nn.Module):
    r"""
    Neural network basis expansion along x- or y- coordinate.

    The coordinates x (or y) are scaled to lie within [0, 1] using the
    [min, max] attributes, as

    $$x' = (x - min) / (max - min)$$

    A nonlinear basis expansion is defined over x' as

    $$B(x') = N(x')$$

    where N is a neural network.

    Notes:
        - Neural network weights are learnable.
        - This basis expansion can directly be used a TorchForce module.

    Attributes:
        min (float): Min x-/y-value for scaling.
        max (float): Max x-/y-value for scaling.
        axis (str): 'x' or 'y' (default='x').
        layers (torch.nn.ModuleList): List of neural layers.

    Args:
        min (float): Min x-/y-value for scaling.
        max (float): Max x-/y-value for scaling.
        axis (str): 'x' or 'y' (default='x').
        hidden_layer_sizes (list): List of neural network hidden layer sizes.
        act (torch.nn.Module): Activation (default=torch.nn.ReLU).
    """
    def __init__(self, min, max, axis='x', hidden_layer_sizes=[], act=nn.ReLU):
        super().__init__()

        self.min = min
        self.max = max
        self.axis = axis

        self.layers = nn.ModuleList()

        if len(hidden_layer_sizes) == 0:
            raise ValueError("NNBasis needs at least one hidden layer")

        # Input layer
        self.layers.append(nn.Linear(1, hidden_layer_sizes[0]))

        # Middle layers
        for lidx in range(1, len(hidden_layer_sizes)):
            self.layers.append(nn.Linear(hidden_layer_sizes[lidx - 1], hidden_layer_sizes[lidx]))
            self.layers.append(act())

        # Output layer
        self.layers.append(nn.Linear(hidden_layer_sizes[-1], 1))

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


################################################################################
# Basis expansions along a 1-dimensional CV in 2D space
################################################################################


class LegendreBasis2DRadialCV(torch.nn.Module):
    r"""
    Legendre polynomial basis expansion along a radial collective variable (CV) defined in the neighborhood
    of points (x_min, y_min) and (x_max, y_max).

    A point $(x, y)$ is mapped to a CV $s$ as

    $$s = ((x - x_min) ** 2 + (y - y_min) ** 2) / ((x_max - x_min) ** 2 + (y_max - y_min) ** 2)$$

    and then scaled to lie within [-1, 1] as

    $$s' = (s - 1/2) / (1/2)$$

    A legendre basis expansion is defined over s', as

    $$B(s') = \sum{i=0}^{d} w_i P_i(s')$$

    where P_i is the legendre polynomial of order i, w_i is its coefficient in
    the expansion, and d is the degree of the expansion.

    Notes:
        - Weights w_i are learnable as self.weights is a torch Parameter.
        - This basis expansion can directly be used a TorchForce module.

    Attributes:
        degree (int): Degree of basis.
        x_min (float): Start x-coordinate.
        y_min (float): Start y-coordinate.
        x_max (float): End x-coordinate.
        y_max (float): End y-coordinate.
        weights (torch.nn.Parameter): Legendre polynomial coefficients (array len = degree).

    Args:
        degree (int): Degree of basis.
        x_min (float): Start x-coordinate.
        y_min (float): Start y-coordinate.
        x_max (float): End x-coordinate.
        y_max (float): End y-coordinate.
        weights (numpy.ndarray): Legendre polynomial coefficients (array len = degree).
    """
    def __init__(self, degree, x_min, y_min, x_max, y_max, weights=None):
        super().__init__()
        self.degree = degree
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

        if weights is not None:
            weights_tensor = torch.from_numpy(weights).type(torch.DoubleTensor)
        else:
            weights_tensor = torch.rand(degree).type(torch.DoubleTensor)
        self.weights = nn.Parameter(weights_tensor)

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
        # Compute string position
        x = positions[:, 0]
        y = positions[:, 1]

        # Compute radial s in [0, 1]
        s = ((x - self.x_min) ** 2 + (y - self.y_min) ** 2) / ((self.x_max - self.x_min) ** 2 + (self.y_max - self.y_min) ** 2)

        # Scale to [-1, 1]
        s = (s - 0.5) / 0.5

        # Clamp to prevent s from going outside Legendre polynomial domain
        s = torch.clamp(s, min=-1, max=1)

        # Apply legendre expansion bias
        bias = torch.zeros_like(s)
        for i in range(self.degree):
            bias += self.weights[i] * self.legendre_polynomial(s, i)
        return bias


class LegendreBasis2DPathCV(torch.nn.Module):
    r"""
    Legendre polynomial basis expansion along a path collective variable (CV) 
    defined using the points $x_i, y_i$, which represent a path in 2D space.

    For a point $(x, y)$ the CVs $s$ and $z$, which measure the location of
    the point parallel to the path and perpendicular to the path respectively, 
    are computed as

    $$s = $$

    $$z = $$

    The CV $s$ is scaled to lie within [-1, 1] as

    $$s' = (s - 1/2) / (1/2)$$

    A legendre basis expansion is defined over s', as

    $$B(s') = \sum{i=0}^{d} w_i P_i(s')$$

    where P_i is the legendre polynomial of order i, w_i is its coefficient in
    the expansion, and d is the degree of the expansion.

    A harmonic bias is defined to restrict

    This basis expansion can directly be used a TorchForce module.

    Attributes:
        degree (int): Degree of basis.
        x_min (float): Start x-coordinate.
        y_min (float): Start y-coordinate.
        x_max (float): End x-coordinate.
        y_max (float): End y-coordinate.
        weights (numpy.ndarray): Legendre polynomial coefficients (array len = degree).
    """
    pass
