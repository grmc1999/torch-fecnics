import torch

# Import fenics and override necessary data structures with fenics_adjoint
import os
import sys
sys.path.append("..")
from torch_fenics.torch_fenics import *
from firedrake.adjoint import *
import firedrake as fd

import torch_fenics

# Declare the FEniCS model corresponding to solving the Poisson equation
# with variable source term and boundary value
class Poisson(torch_fenics.FEniCSModule):
    # Construct variables which can be reused in the constructor
    def __init__(self):
        # Call super constructor
        super().__init__()

        # Create function space
        mesh = fd.UnitIntervalMesh(20)
        self.V = fd.FunctionSpace(mesh, 'P', 1)

        # Create trial and test functions
        u = fd.TrialFunction(self.V)
        self.v = fd.TestFunction(self.V)

        # Construct bilinear form
        self.a = fd.inner(fd.grad(u), fd.grad(self.v)) * fd.dx

    def solve(self, f, g):
        # Construct linear form
        L = f * self.v * fd.dx

        # Construct boundary condition
        bc = fd.DirichletBC(self.V, g, 'on_boundary')

        # Solve the Poisson equation
        u = fd.Function(self.V)
        fd.solve(self.a == L, u, bc)

        # Return the solution
        return u

    def input_templates(self):
        # Declare templates for the inputs to Poisson.solve
        return fd.Constant(0), fd.Constant(0)


if __name__ == '__main__':
    # Construct the FEniCS model
    poisson = Poisson()

    # Create N sets of input
    N = 10
    f = torch.rand(N, 1, requires_grad=True, dtype=torch.float64)
    g = torch.rand(N, 1, requires_grad=True, dtype=torch.float64)

    # Solve the Poisson equation N times
    u = poisson(f, g)

    # Construct functional
    J = u.sum()

    # Execute the backward pass
    J.backward()

    # Extract gradients
    dJdf = f.grad
    dJdg = g.grad
