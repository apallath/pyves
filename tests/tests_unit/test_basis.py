"""
Unit tests for ves.basis
"""
import numpy as np
from scipy.special import legendre
import torch

from ves.basis import LegendreBasis_x


def test_legendre_polynomials():
    """
    Benchmarks the outputs of ves.bias.legendre_polynomial against scipy.special.legendre.
    """
    for degree in range(5):
        x = torch.linspace(-1, 1, 1000)
        lves = LegendreBasis_x.legendre_polynomial(x, degree).detach().cpu().numpy()
        lsci = legendre(degree)(x.cpu().numpy())
        assert(np.allclose(lves, lsci))


def test_LegendreBasis():
    """
    Checks whether the LegendreBasis torch module works.
    """
    pass


def test_NNBasis():
    """
    Checks whether the NNBasis torch module works.
    """
    pass
