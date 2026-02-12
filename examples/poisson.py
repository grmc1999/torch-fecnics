import torch

# Import fenics and override necessary data structures with fenics_adjoint
import os
import sys
import dolfinx
from dolfinx import fem,mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
sys.path.append("..")
from torch_fenics.torch_fenics import *

import torch_fenics

# Declare the FEniCS model corresponding to solving the Poisson equation
# with variable source term and boundary value
class Poisson(torch_fenics.FEniCSModule):
    # Construct variables which can be reused in the constructor
    def __init__(self):
        # Call super constructor
        super().__init__()

        # Create function space
        #mesh = UnitIntervalMesh(20)
        self.domain = mesh.create_unit_interval(MPI.COMM_WORLD,10)
        self.V = fem.functionspace(self.domain, ('P', 1))

        # Create trial and test functions
        u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)

        # Construct bilinear form
        self.a = ufl.inner(ufl.grad(u), ufl.grad(self.v)) * ufl.dx

    def make_boundary(self,g):
        tdim = self.domain.topology.dim
        fdim = tdim - 1
        self.domain.topology.create_connectivity(fdim, tdim)
        boundary_facets = mesh.exterior_facet_indices(self.domain.topology)
        boundary_dofs = fem.locate_dofs_topological(self.V, fdim, boundary_facets)
        bc_exp = fem.Expression(g,self.V.element.interpolation_points())
        self.uD = fem.Function(self.V)
        self.uD.interpolate(bc_exp)
        self.bc = fem.dirichletbc(self.uD, boundary_dofs)

    def solve(self, f, g):
        # Construct linear form
        L = f * self.v * ufl.dx

        # Construct boundary condition
        self.make_boundary(g)

        # Solve the Poisson equation
        u = fem.functionspace(self.V)
        #solve(self.a == L, u, bc)
        P1 = LinearProblem(
        self.a,
        L,
        bcs=[self.bc],
        petsc_options={
            "ksp_type":"preonly",
            "pc_type":"lu"
            },
        #petsc_options_prefix="Poisson"
        )
        u1 = P1.solve()

        # Return the solution
        return u1

    def input_templates(self):
        # Declare templates for the inputs to Poisson.solve
        return fem.Constant(self.domain,0.0), fem.Constant(self.domain,0.0)


if __name__ == '__main__':
    # Construct the FEniCS model
    poisson = Poisson()

    # Create N sets of input
    N = 10
    f = torch.rand(1, requires_grad=True, dtype=torch.float64)
    g = torch.rand(1, requires_grad=True, dtype=torch.float64)

    # Solve the Poisson equation N times
    u = poisson(f, g)

    # Construct functional
    J = u.sum()

    # Execute the backward pass
    J.backward()

    # Extract gradients
    dJdf = f.grad
    dJdg = g.grad
