"""
Classes defining basis expansions.
"""
import torch
import torch.nn as nn

from deprecated import deprecated

torch.set_default_tensor_type(torch.DoubleTensor)

################################################################################
# 1D basis expansions: along x or y axes.
################################################################################


class LegendreBasis1D(torch.nn.Module):
    r"""
    Legendre polynomial basis expansion along x- or y- coordinate.

    The coordinates x (or y) are scaled to lie within [-1, 1] using the
    [min, max] attributes, as

    $$x' = \frac{x - (min + max) / 2}{(max - min) / 2}$$

    A legendre basis expansion is defined over $x'$, as

    $$B(x') = \sum_{i=0}^{d} w_i P_i(x')$$

    where $P_i$ is the legendre polynomial of degree $i$, $w_i$ is its coefficient in
    the expansion, and $d$ is the degree of the expansion.

    Notes:
        - Weights $w_i$ are learnable as `self.weights` is a torch Parameter.
        - This basis expansion can directly be used a TorchForce module.

    Attributes:
        degree (int): Degree of expansion.
        min (float): Min x-/y-value for scaling.
        max (float): Max x-/y-value for scaling.
        axis (str): 'x' or 'y' (default='x').
        weights (torch.nn.Parameter): Legendre polynomial coefficients (array len = degree + 1).
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
            weights_tensor = torch.rand(degree + 1).type(torch.DoubleTensor)
        self.weights = nn.Parameter(weights_tensor)

    @classmethod
    @deprecated("""Use this method only if you need to compute a single Legendre polynomial. 
    If you need to compute a Legendre basis expansion, use legendre_polynomial_expansion instead""")
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

    @classmethod
    def legendre_polynomial_expansion(cls, x, degree: int, weights: torch.Tensor) -> torch.Tensor:
        r"""
        Computes a legendre polynomial expansion of degree $d$ using dynamic programming
        and the Bonnet's recursion formula:

        $$(n + 1) P_{n+1}(x) = (2n + 1) x P_n(x) - nP_{n-1}(x)$$

        $$y = \sum_{n=1}^{d} w_n P_n(x)$$

        Important:
            `weights` must be of length (`degree` + 1).
        """
        ones_list = x.size(0) * [1.0]
        P_n_minus = torch.tensor(ones_list, requires_grad=True).type(x.type())

        out = weights[0] * P_n_minus
        if degree == 0:
            return out

        P_n = x
        out += weights[1] * P_n
        if degree == 1:
            return out

        for n in range(1, degree):
            P_n_plus = ((2 * n + 1) * x * P_n - n * P_n_minus) / (n + 1)

            # Replace
            P_n_minus = P_n
            P_n = P_n_plus

            # Add to expansion
            out += weights[n + 1] * P_n

        return out

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

        # bias = torch.zeros_like(x)
        # for i in range(self.degree + 1):
        #     bias += self.weights[i] * self.legendre_polynomial(x, i)

        bias = self.legendre_polynomial_expansion(x, self.degree, self.weights)
        return bias


class NNBasis1D(nn.Module):
    r"""
    Neural network basis expansion along x- or y- coordinate.

    The coordinates x (or y) are scaled to lie within [0, 1] using the
    [min, max] attributes, as

    $$x' = (x - min) / (max - min)$$

    A nonlinear basis expansion is defined over $x'$ as

    $$B(x') = N(x')$$

    where $N$ is a neural network.

    Notes:
        - Neural network weights are learnable.
        - This basis expansion can directly be used a TorchForce module.

    Attributes:
        min (float): Min x-/y-value for scaling.
        max (float): Max x-/y-value for scaling.
        axis (str): 'x' or 'y' (default='x').
        layers (torch.nn.ModuleList): List of neural layers.
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

@deprecated("Use LegendreBasis2DPathCV instead.")
class LegendreBasis2DRadialCV(torch.nn.Module):
    r"""
    Legendre polynomial basis expansion along a radial collective variable (CV) defined in the neighborhood
    of points $(x_min, y_min)$ and $(x_max, y_max)$.

    A point $(x, y)$ is mapped to a CV $s$ as

    $$s = \frac{(x - x_{min})^2 + (y - y_{min})^2}{(x_{max} - x_{min})^2 + (y_{max} - y_{min})^2}$$

    and then scaled to lie within [-1, 1] as

    $$s' = (s - 1/2) / (1/2)$$

    A legendre basis expansion is defined over $s'$, as

    $$B(s') = \sum_{i=0}^{d} w_i P_i(s')$$

    where $P_i$ is the legendre polynomial of order $i$, $w_i$ is its coefficient in
    the expansion, and $d$ is the degree of the expansion.

    Notes:
        - Weights $w_i$ are learnable as `self.weights` is a torch Parameter.
        - This basis expansion can directly be used a TorchForce module.

    Attributes:
        degree (int): Degree of basis.
        x_min (float): Start x-coordinate.
        y_min (float): Start y-coordinate.
        x_max (float): End x-coordinate.
        y_max (float): End y-coordinate.
        weights (torch.nn.Parameter): Legendre polynomial coefficients (array len = degree + 1).
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
            weights_tensor = torch.rand(degree + 1).type(torch.DoubleTensor)
        self.weights = nn.Parameter(weights_tensor)

    @classmethod
    @deprecated("""Use this method only if you need to compute a single Legendre polynomial. 
    If you need to compute a Legendre basis expansion, use legendre_polynomial_expansion instead""")
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

    @classmethod
    def legendre_polynomial_expansion(cls, x, degree: int, weights: torch.Tensor) -> torch.Tensor:
        r"""
        Computes a legendre polynomial expansion of degree $d$ using dynamic programming
        and the Bonnet's recursion formula:

        $$(n + 1) P_{n+1}(x) = (2n + 1) x P_n(x) - nP_{n-1}(x)$$

        $$y = \sum_{n=1}^{d} w_n P_n(x)$$

        Important:
            `weights` must be of length (`degree` + 1).
        """
        ones_list = x.size(0) * [1.0]
        P_n_minus = torch.tensor(ones_list, requires_grad=True).type(x.type())

        out = weights[0] * P_n_minus
        if degree == 0:
            return out

        P_n = x
        out += weights[1] * P_n
        if degree == 1:
            return out

        for n in range(1, degree):
            P_n_plus = ((2 * n + 1) * x * P_n - n * P_n_minus) / (n + 1)

            # Replace
            P_n_minus = P_n
            P_n = P_n_plus

            # Add to expansion
            out += weights[n + 1] * P_n

        return out

    def forward(self, positions):
        """The forward method returns the energy computed from positions.

        Args:
            positions : torch.Tensor with shape (1, 3)
                positions[0, k] is the position (in nanometers) of spatial dimension k of particle 0

        Returns:
            potential : torch.Scalar
                The potential energy (in kJ/mol)
        """
        x = positions[:, 0]
        y = positions[:, 1]

        # Compute radial s in [0, 1]
        s = ((x - self.x_min) ** 2 + (y - self.y_min) ** 2) / ((self.x_max - self.x_min) ** 2 + (self.y_max - self.y_min) ** 2)

        # Scale to [-1, 1]
        s = (s - 0.5) / 0.5

        # Clamp to prevent s from going outside Legendre polynomial domain
        s = torch.clamp(s, min=-1, max=1)

        # Apply legendre expansion bias
        
        # bias = torch.zeros_like(s)
        # for i in range(self.degree):
        #     bias += self.weights[i] * self.legendre_polynomial(s, i)

        bias = self.legendre_polynomial_expansion(s, self.degree, self.weights)
        return bias


class LegendreBasis2DPathCV(torch.nn.Module):
    r"""
    Legendre polynomial basis expansion along a path collective variable (CV) 
    defined using the points $x_i, y_i$, which represent a path in 2D space.

    For a point $(x, y)$ the CVs $s$ and $z$, which measure the location of
    the point parallel to the path and perpendicular to the path respectively, 
    are computed as

    $$s = \frac{1}{N} \frac{\sum_{i=0}^{N-1} (i + 1)\ e^{-\lambda [(x - x_i) ^ 2 + (y - y_i) ^ 2]}}{\sum_{i=0}^{N-1} e^{-\lambda [(x - x_i) ^ 2 + (y - y_i) ^ 2]}}$$

    $$z = -\frac{1}{\lambda} \ln (\sum_{i=0}^{N-1} e^{-\lambda [(x - x_i) ^ 2 + (y - y_i) ^ 2]})$$

    The CV $s$ is scaled to lie within [-1, 1] as

    $$s' = (s - 1/2) / (1/2)$$

    A legendre basis expansion is defined over $s'$, as

    $$B(s') = \sum_{i=0}^{d} w_i P_i(s')$$

    where $P_i$ is the legendre polynomial of order $i$, $w_i$ is its coefficient in
    the expansion, and $d$ is the degree of the expansion.

    A restraining linear bias is defined to restrict sampling to regions near the path, as

    $$U(z) = \phi z$$

    This basis expansion can directly be used a TorchForce module.

    Attributes:
        degree (int): Degree of basis.
        Npath (int): Number of images along the path.
        x_i (torch.DoubleTensor): x-coordinates of images along the path.
        y_i (torch.DoubleTensor): y-coordinates of images along the path.
        lam (float): Value of $\lambda$ (choose a sufficiently large value).
        weights (torch.nn.Parameter): Legendre polynomial coefficients of expansion along $s$ (array len = degree + 1).
        phi (torch.DoubleTensor): Strength of restraining potential along $z$.
    """
    def __init__(self, degree, x_i, y_i, lam, weights=None, phi=0):
        super().__init__()
        self.degree = degree
        assert(len(x_i) == len(y_i))
        self.Npath = len(x_i)
        self.x_i = torch.from_numpy(x_i).type(torch.DoubleTensor)
        self.y_i = torch.from_numpy(y_i).type(torch.DoubleTensor)
        self.lam = lam

        if weights is not None:
            weights_tensor = torch.from_numpy(weights).type(torch.DoubleTensor)
        else:
            weights_tensor = torch.rand(degree + 1).type(torch.DoubleTensor)
        self.weights = nn.Parameter(weights_tensor)

        self.phi = phi

    @classmethod
    @deprecated("""Use this method only if you need to compute a single Legendre polynomial. 
    If you need to compute a Legendre basis expansion, use legendre_polynomial_expansion instead""")
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

    @classmethod
    def legendre_polynomial_expansion(cls, x, degree: int, weights: torch.Tensor) -> torch.Tensor:
        r"""
        Computes a legendre polynomial expansion of degree $d$ using dynamic programming
        and the Bonnet's recursion formula:

        $$(n + 1) P_{n+1}(x) = (2n + 1) x P_n(x) - nP_{n-1}(x)$$

        $$y = \sum_{n=1}^{d} w_n P_n(x)$$

        Important:
            `weights` must be of length (`degree` + 1).
        """
        ones_list = x.size(0) * [1.0]
        P_n_minus = torch.tensor(ones_list, requires_grad=True).type(x.type())

        out = weights[0] * P_n_minus
        if degree == 0:
            return out

        P_n = x
        out += weights[1] * P_n
        if degree == 1:
            return out

        for n in range(1, degree):
            P_n_plus = ((2 * n + 1) * x * P_n - n * P_n_minus) / (n + 1)

            # Replace
            P_n_minus = P_n
            P_n = P_n_plus

            # Add to expansion
            out += weights[n + 1] * P_n

        return out

    def forward(self, positions):
        """The forward method returns the energy computed from positions.

        Args:
            positions : torch.Tensor with shape (1, 3)
                positions[0, k] is the position (in nanometers) of spatial dimension k of particle 0

        Returns:
            potential : torch.Scalar
                The potential energy (in kJ/mol)
        """
        x = positions[:, 0]
        y = positions[:, 1]

        # Compute parallel distance OP s
        ivals = torch.arange(1, self.Npath + 1).type(x.type())
        s = 1 / self.Npath * torch.exp(torch.logsumexp(-self.lam * ((x.unsqueeze(0) - self.x_i.unsqueeze(-1)) ** 2 + (y.unsqueeze(0) - self.y_i.unsqueeze(-1)) ** 2) + torch.log(ivals).unsqueeze(-1), 0) 
                                       - torch.logsumexp(-self.lam * ((x.unsqueeze(0) - self.x_i.unsqueeze(-1)) ** 2 + (y.unsqueeze(0) - self.y_i.unsqueeze(-1)) ** 2), 0))

        # Compute perpendicular distance OP z
        z = -1 / self.lam * torch.logsumexp(-self.lam * ((x.unsqueeze(0) - self.x_i.unsqueeze(-1)) ** 2 + (y.unsqueeze(0) - self.y_i.unsqueeze(-1)) ** 2), 0)

        # Scale to [-1, 1]
        s = (s - 0.5) / 0.5

        # Clamp to prevent s from going outside Legendre polynomial domain
        s = torch.clamp(s, min=-1, max=1)

        # Apply legendre expansion bias to s

        # bias = torch.zeros_like(s)
        # for i in range(self.degree):
        #     bias += self.weights[i] * self.legendre_polynomial(s, i)

        bias = self.legendre_polynomial_expansion(s, self.degree, self.weights)

        # Apply harmonic bias to z
        bias += self.phi * z

        return bias
