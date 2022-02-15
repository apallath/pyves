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
    Distribution is uniform over interval [-1, 1]. Outside of this, it is zero.
    """
    def __init__(self, mesh):
        super().__init__()
        self._x = np.linspace(-1, 1, mesh)
        self._p = 1 / 2 * np.ones(self._x.shape)
