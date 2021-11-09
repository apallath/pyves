"""
Classes defining potential energy surfaces.
"""
from simtk.openmm import openmm


class Potential2D(openmm.CustomExternalForce):
    """
    Abstract class defining basic 2D potential energy behavior and plotting
    functions.
    """
    def __init__(self):
        # Apply restraining potential along z direction
        # Child classes will add terms for x and y and initialize this force expression
        self.force = "1000 * z^2"

    def potential(self, x: float, y: float):
        """
        Computes the potential at a given point (x, y).
        """
        # Child classes will implement this method.
        pass
