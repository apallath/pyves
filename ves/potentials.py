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
    1D single gaussian well potential.
    """
    def __init__(self):
        self.force = """ """

        super().__init__()

    def potential(self, x):
        pass


class DoubleWellPotential1D(Potential1D):
    """
    1D double gaussian well potential.
    """
    pass


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
    pass


class DoubleWellPotential2D(Potential2D):
    pass


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
    pass
