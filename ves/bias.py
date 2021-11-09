"""
Classes defining VES bias potentials
"""


class Bias:
    """
    Abstract class defining methods for VES bias to implement.
    """
    def __init__(self):
        pass

    def update(self, traj):
        pass

    @property
    def force(self):
        raise NotImplementedError("Force not implemented.")
        return None


class BasisSetExpansionBias(Bias):
    pass


class NeuralNetworkBias(Bias):
    pass
