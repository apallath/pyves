"""
Classes defining potential energy surfaces.
"""
import numpy as np
from openmm import openmm

################################################################################
# 2D potentials
################################################################################


class Potential1D(openmm.CustomExternalForce):
    """
    Abstract class defining basic 1D potential behavior.

    A harmonic restraining potential of magnitude 1000 kJ/mol is applied on the
    y and z coordinates about y=0 and z=0.

    Note:
        Child classes must call super.__init__() only after initializing the force
        attribute in the x variable.

    Attributes:
        force (str): `OpenMM-compatible custom force expression`_.

    .. _OpenMM-compatible custom force expression:
       http://docs.openmm.org/latest/userguide/theory/03_custom_forces.html#writing-custom-expressions
    """
    def __init__(self):
        # Apply restraining potential along z direction
        # Child classes will add terms for x and y and initialize this force expression
        self.force += " + 1000 * y^2 + 1000 * z^2"

        # Print force expression
        print("[Potential] Initializing potential with expression:\n" + self.force)

        # Initialize force expression
        super().__init__(self.force)

    def potential(self, x: float):
        """
        Computes the potential at a given point x.

        Args:
            x (float): Point to compute potential at.

        Returns:
            V (float): Value of potential at x.
        """
        # Child classes will implement this method.
        raise NotImplementedError()


class SingleWellPotential1D(Potential1D):
    """
    1D single well potential.
    """
    def __init__(self):
        self.force = '''x^2'''

        super().__init__()

    def potential(self, x):
        return x ** 2


class DoubleWellPotential1D(Potential1D):
    """
    1D double well potential.
    """
    def __init__(self):
        self.force = '''x^4 - 4 * x^2 + 0.7 * x'''

        super().__init__()

    def potential(self, x):
        return x ** 4 - 4 * x ** 2 + 0.7 * x


################################################################################
# 2D potentials
################################################################################


class Potential2D(openmm.CustomExternalForce):
    """
    Abstract class defining basic 2D potential behavior.

    A harmonic restraining potential of magnitude 1000 kJ/mol is applied on the
    z coordinates about z=0.

    Note:
        Child classes must call super.__init__() only after initializing the force
        attribute in x and y variables.

    Attributes:
        force (str): `OpenMM-compatible custom force expression`_.

    .. _OpenMM-compatible custom force expression:
       http://docs.openmm.org/latest/userguide/theory/03_custom_forces.html#writing-custom-expressions
    """
    def __init__(self):
        # Apply restraining potential along z direction
        # Child classes will add terms for x and y and initialize this force expression
        self.force += " + 1000 * z^2"

        # Print force expression
        print("[Potential] Initializing potential with expression:\n" + self.force)

        # Initialize force expression
        super().__init__(self.force)

    def potential(self, x: float, y: float):
        """
        Computes the potential at a given point (x, y).

        Args:
            x (float): x-coordinate of the point to compute potential at.
            y (float): y-coordinate of the point to compute potential at.

        Returns:
            V (float): Value of potential at (x, y).
        """
        # Child classes will implement this method.
        raise NotImplementedError()


class SingleWellPotential2D(Potential2D):
    r"""
    Single well potential.

    $$U(x, y) = x^2 + y^2$$
    """
    def __init__(self):
        self.force = '''x^2 + y^2'''

        super().__init__()

    def potential(self, x, y):
        """Computes the single well potential at a given point (x, y)."""
        return x ** 2 + y ** 2


class DoubleWellPotential2D(Potential2D):
    r"""
    Double well potential.

    $$U(x, y) = x^4 - 4 x^2 + 0.7 x + 5 y^2$$
    """
    def __init__(self):
        self.force = '''x^4 - 4 * x^2 + 0.7 * x + 5 * y^2'''

        super().__init__()

    def potential(self, x, y):
        """Computes the double well potential at a given point (x, y)."""
        return x ** 4 - 4 * x ** 2 + 0.7 * x + 5 * y ** 2


class SlipBondPotential2D(Potential2D):
    r"""
    2-basin slip bond potential.

    $$U(x, y) = \left( \left(\frac{(y - y\_0)^2}{y\_scale} - y\_shift \right)^2 + \frac{(x - y - xy\_0)^2}{xy\_scale} \right)$$
    """
    def __init__(self, force_x=0, force_y=0, y_0=1, y_scale=5, y_shift=4, xy_0=0, xy_scale=2):
        self.force_x = force_x
        self.force_y = force_y
        self.y_0 = y_0
        self.y_scale = y_scale
        self.y_shift = y_shift
        self.xy_0 = xy_0
        self.xy_scale = xy_scale

        constvals = {"force_x": self.force_x,
                     "force_y": self.force_y,
                     "y_0": self.y_0,
                     "y_scale": self.y_scale,
                     "y_shift": self.y_shift,
                     "xy_0": self.xy_0,
                     "xy_scale": self.xy_scale}

        self.force = '''((y - {y_0})^2 / {y_scale} - {y_shift})^2 + (x - y - {xy_0})^2 / {xy_scale} - {force_x} * x - {force_y} * y'''.format(**constvals)

        super().__init__()

    def potential(self, x, y):
        """Computes the slip bond potential at a given point (x, y)."""
        return ((y - self.y_0) ** 2 / self.y_scale - self.y_shift) ** 2 + (x - y - self.xy_0) ** 2 / self.xy_scale - self.force_x * x - self.force_y * y


class CatchBondPotential2D(Potential2D):
    r"""
    3-basin catch bond potential (slip bond potential with an extra harmonic basin).

    $$U(x, y) = -\ln \left[ e^{-\left( \left(\frac{(y - y_0)^2}{y_{scale}} - y_{shift} \right)^2 + \frac{(x - y - xy_{shift})^2}{2} \right) }
    + e^-{(x - gx_0)^2 / gx_scale + (y - gy_0)^2 / gy_scale} \right]$$
    """
    def __init__(self, force_x=0, force_y=0, y_0=1, y_scale=5, y_shift=4, xy_0=0, xy_scale=2, gx_0=2, gx_scale=0.5, gy_0=-2.5, gy_scale=0.25):
        self.force_x = force_x
        self.force_y = force_y
        self.y_0 = y_0
        self.y_scale = y_scale
        self.y_shift = y_shift
        self.xy_0 = xy_0
        self.xy_scale = xy_scale

        # Harmonic basin parameters
        self.gx_0 = gx_0
        self.gx_scale = gx_scale
        self.gy_0 = gy_0
        self.gy_scale = gy_scale

        constvals = {"force_x": self.force_x,
                     "force_y": self.force_y,
                     "y_0": self.y_0,
                     "y_scale": self.y_scale,
                     "y_shift": self.y_shift,
                     "xy_0": self.xy_0,
                     "xy_scale": self.xy_scale,
                     "gx_0": self.gx_0,
                     "gx_scale": self.gx_scale,
                     "gy_0": self.gy_0,
                     "gy_scale": self.gy_scale}

        self.force = '''-log(exp(-(((y - {y_0})^2 / {y_scale} - {y_shift})^2 + (x - y - {xy_0})^2 / {xy_scale}) ) + exp(-( (x - {gx_0})^2 / {gx_scale} + (y - {gy_0})^2 / {gy_scale} )) ) - {force_x} * x - {force_y} * y'''.format(**constvals)

        super().__init__()

    def potential(self, x, y):
        """Computes the catch bond potential at a given point (x, y)."""
        return -np.log(np.exp(-(((y - self.y_0) ** 2 / self.y_scale - self.y_shift) ** 2 + (x - y - self.xy_0) ** 2 / self.xy_scale)) +
                       np.exp(-((x - self.gx_0) ** 2 / self.gx_scale + (y - self.gy_0) ** 2 / self.gy_scale))) - self.force_x * x - self.force_y * y


class SzaboBerezhkovskiiPotential(Potential2D):
    """
    2D Szabo-Berezhkovskii potential.
    """
    # Constants that define the potential function
    x0 = 2.2
    omega2 = 4.0
    Omega2 = 1.01 * omega2
    Delta = omega2 * x0 ** 2 / 4.0

    def __init__(self):
        # Look up Szabo-Berezhkovskii potential formula for details
        constvals = {"x0": self.x0,
                     "omega2": self.omega2,
                     "Omega2": self.Omega2,
                     "Delta": self.Delta}

        self.force = '''{Omega2} * 0.5 * (x - y)^2'''.format(**constvals)
        self.force += ''' + (select(step(x + 0.5 * {x0}), select(step(x - 0.5 * {x0}), -{Delta} + {omega2} * 0.5 * (x - {x0})^2, -{omega2} * 0.5 * x^2), -{Delta} + {omega2} * 0.5 * (x + {x0})^2))'''.format(**constvals)

        super().__init__()

    def potential(self, x, y):
        """Computes the Szabo-Berezhkovskii potential at a given point (x, y)."""
        Ux = np.piecewise(x,
                          [x <= -self.x0 / 2,
                           np.logical_and(x > -self.x0 / 2, x < self.x0 / 2),
                           x >= self.x0 / 2],
                          [lambda x: -self.Delta + self.omega2 * (x + self.x0) ** 2 / 2.0,
                           lambda x: -self.omega2 * x ** 2 / 2.0,
                           lambda x: -self.Delta + self.omega2 * (x - self.x0) ** 2 / 2.0])
        return (Ux + self.Omega2 * (x - y) ** 2 / 2.0)


class MullerBrownPotential(Potential2D):
    """
    2D Muller-Brown potential.
    """
    a = [-1, -1, -6.5, 0.7]
    b = [0, 0, 11, 0.6]
    c = [-10, -10, -6.5, 0.7]
    A = [-200, -100, -170, 15]
    x_bar = [1, 0, -0.5, -1]
    y_bar = [0, 0.5, 1.5, 1]

    def __init__(self):
        for i in range(4):
            fmt = dict(a=self.a[i], b=self.b[i], c=self.c[i], A=self.A[i], x_bar=self.x_bar[i], y_bar=self.y_bar[i])
            if i == 0:
                self.force = '''{A} * exp({a} * (x - {x_bar})^2 + {b} * (x - {x_bar}) * (y - {y_bar}) + {c} * (y - {y_bar})^2)'''.format(**fmt)
            else:
                self.force += ''' + {A} * exp({a} * (x - {x_bar})^2 + {b} * (x - {x_bar}) * (y - {y_bar}) + {c} * (y - {y_bar})^2)'''.format(**fmt)

        super().__init__()

    def potential(self, x, y):
        """Compute the Muller-Brown potential at a given point (x, y)."""
        value = 0
        for i in range(4):
            value += self.A[i] * np.exp(self.a[i] * (x - self.x_bar[i])**2 +
                                        self.b[i] * (x - self.x_bar[i]) * (y - self.y_bar[i]) + self.c[i] * (y - self.y_bar[i])**2)
        return value


class WolfeQuappPotential(Potential2D):
    """
    2D Wolfe-Quapp potential
    """
    pass
