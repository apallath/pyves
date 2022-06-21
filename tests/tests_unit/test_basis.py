"""
Unit tests for ves.basis
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import legendre
import torch

from ves.basis import LegendreBasis1D

def test_legendre_polynomials():
    """
    Benchmarks ves.basis.LegendreBasis1D.legendre_polynomial against scipy.special.legendre.
    """
    fig, ax = plt.subplots(dpi=200)

    for degree in range(5):
        x = torch.linspace(-1, 1, 1000)
        lves = LegendreBasis1D.legendre_polynomial(x, degree).detach().cpu().numpy()
        lsci = legendre(degree)(x.cpu().numpy())
        assert(np.allclose(lves, lsci))
        ax.plot(x.cpu().numpy(), lsci)
        ax.plot(x.cpu().numpy(), lves, '--')

    plt.savefig("legendre_compare.png")


def test_legendre_polynomial_expansion():
    """
    Benchmarks ves.basis.LegendreBasis1D.legendre_polynomial_expansion against ves.basis.LegendreBasis1D.legendre_polynomial
    """
    fig, ax = plt.subplots(dpi=200)

    x = torch.linspace(-1, 1, 1000)
    weights = torch.rand(5 + 1)

    out_bench = torch.zeros_like(x)
    for i in range(5 + 1):
        out_bench += weights[i] * LegendreBasis1D.legendre_polynomial(x, i)
        
    out_bench = out_bench.detach().cpu().numpy()
        
    out_test = LegendreBasis1D.legendre_polynomial_expansion(x, 5, weights).detach().cpu().numpy()
    
    assert(np.allclose(out_bench, out_test))


def test_grad_legendre_polynomials():
    """
    Benchmarks gradients (derivatives) of ves.basis.LegendreBasis_x against those of scipy.special.legendre.
    """
    fig, ax = plt.subplots(dpi=200)

    for degree in range(1, 5):
        x = torch.linspace(-1, 1, 1000, requires_grad=True)

        lves = LegendreBasis1D.legendre_polynomial(x, degree)
        lves_grad = torch.autograd.grad(lves, x, grad_outputs=torch.ones_like(lves), create_graph=True, allow_unused=True)[0].detach().cpu().numpy()

        lsci = legendre(degree)
        lsci_grad = lsci.deriv()(x.detach().cpu().numpy())

        assert(np.allclose(lves_grad, lsci_grad))

        ax.plot(x.detach().cpu().numpy(), lsci_grad)
        ax.plot(x.detach().cpu().numpy(), lves_grad, '--')
        
    plt.savefig("grad_legendre_compare.png")

def test_grad_legendre_polynomial_expansion():
    """
    Benchmarks gradients of ves.basis.LegendreBasis1D.legendre_polynomial_expansion against those of
    ves.basis.LegendreBasis1D.legendre_polynomial
    """
    fig, ax = plt.subplots(dpi=200)

    x = torch.linspace(-1, 1, 1000, requires_grad=True)
    weights = torch.rand(5 + 1)

    out_bench = torch.zeros_like(x)
    for i in range(5 + 1):
        out_bench += weights[i] * LegendreBasis1D.legendre_polynomial(x, i)
        
    grad_bench = torch.autograd.grad(out_bench, x, grad_outputs=torch.ones_like(out_bench), create_graph=True, allow_unused=True)[0].detach().cpu().numpy()
        
    out_test = LegendreBasis1D.legendre_polynomial_expansion(x, 5, weights)
    grad_test = torch.autograd.grad(out_bench, x, grad_outputs=torch.ones_like(out_test), create_graph=True, allow_unused=True)[0].detach().cpu().numpy()
    
    assert(np.allclose(grad_bench, grad_test))
