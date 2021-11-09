"""
Classes defining potential energy surfaces.
"""
import numpy as np
from openmm import openmm


class Potential2D(openmm.CustomExternalForce):
    """
    Abstract class defining basic 2D potential energy behavior and plotting
    functions.
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
        """
        # Child classes will implement this method.
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
        """Computes the Szabo-Berezhkovskii potential at a given point (x, y)"""
        Ux = np.piecewise(x,
                          [x <= -self.x0 / 2,
                           np.logical_and(x > -self.x0 / 2, x < self.x0 / 2),
                           x >= self.x0 / 2],
                          [lambda x: -self.Delta + self.omega2 * (x + self.x0) ** 2 / 2.0,
                           lambda x: -self.omega2 * x ** 2 / 2.0,
                           lambda x: -self.Delta + self.omega2 * (x - self.x0) ** 2 / 2.0])
        return (Ux + self.Omega2 * (x - y) ** 2 / 2.0)
