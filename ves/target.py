"""
Class defining VES target free energy landscape
"""
import numpy as np

class Target_x:
    """
    Abstract class defining VES target probability distribution.
    """
    def __init__(self):
        self._x = None
        self._p = None

    @property
    def x(self):
        return self._x

    @property
    def p(self):
        return self._p


class Target_Uniform_HardSwitch_x(Target_x):
    """
    Distribution is uniform over interval [x_min, x_max]. Outside of this, it is zero.
    """
    def __init__(self, x_min, x_max, mesh):
        super().__init__()
        self._x = np.linspace(x_min, x_max, mesh)
        self._p = 1 / (x_max - x_min) * np.ones(self._x.shape)
