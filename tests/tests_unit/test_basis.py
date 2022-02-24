"""
Unit tests for ves.basis
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import legendre
import torch

from ves.basis import LegendreBasis_x


def test_legendre_polynomials():
    """
    Benchmarks ves.basis.LegendreBasis_x against scipy.special.legendre.
    """
    fig, ax = plt.subplots(dpi=200)

    for degree in range(5):
        x = torch.linspace(-1, 1, 1000)
        lves = LegendreBasis_x.legendre_polynomial(x, degree).detach().cpu().numpy()
        lsci = legendre(degree)(x.cpu().numpy())
        assert(np.allclose(lves, lsci))
        ax.plot(x.cpu().numpy(), lsci)
        ax.plot(x.cpu().numpy(), lves, '--')

    plt.savefig("legendre_compare.png")


def test_grad_legendre_polynomials():
    """
    Benchmarks gradients (derivatives) of ves.basis.LegendreBasis_x against those of scipy.special.legendre.
    """
    fig, ax = plt.subplots(dpi=200)

    for degree in range(1, 5):
        x = torch.linspace(-1, 1, 1000, requires_grad=True)

        lves = LegendreBasis_x.legendre_polynomial(x, degree)
        lves_grad = torch.autograd.grad(lves, x, grad_outputs=torch.ones_like(lves), create_graph=True, allow_unused=True)[0].detach().cpu().numpy()

        lsci = legendre(degree)
        lsci_grad = lsci.deriv()(x.detach().cpu().numpy())

        assert(np.allclose(lves_grad, lsci_grad))

        ax.plot(x.detach().cpu().numpy(), lsci_grad)
        ax.plot(x.detach().cpu().numpy(), lves_grad, '--')
        
    plt.savefig("grad_legendre_compare.png")


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
