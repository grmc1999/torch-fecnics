from firedrake import *
from firedrake.adjoint import *

import torch_fenics
import torch


class Poisson(torch_fenics.FiredrakeModule):
    # Construct variables which can be reused in the constructor
    def __init__(self):
        # Call super constructor
        super().__init__()

        # Create function space
        self.mesh = UnitIntervalMesh(20)
        self.V = FunctionSpace(self.mesh, 'P', 1)

        # Create trial and test functions
        u = TrialFunction(self.V)
        self.v = TestFunction(self.V)

        # Construct bilinear form
        self.a = inner(grad(u), grad(self.v)) * dx

    def solve(self, f, g):
        # Construct linear form
        L = f * self.v * dx
        print("pass L")

        # Construct boundary condition
        bc = DirichletBC(self.V, g, 'on_boundary')
        print("pass BC")

        # Solve the Poisson equation
        u = Function(self.V)
        solve(self.a == L, u, bc)
        print("pass solve")

        # Return the solution
        return u

    def input_templates(self):
        # Declare templates for the inputs to Poisson.solve
        #return Constant(0), Constant(0)
        return Function(FunctionSpace(self.mesh,'R',0)),Function(FunctionSpace(self.mesh,'R',0))


if __name__ == '__main__':
    # Construct the FEniCS model
    poisson = Poisson()

    # Create N sets of input
    N = 10
    f = torch.rand(N, 1, requires_grad=True, dtype=torch.float64)
    g = torch.rand(N, 1, requires_grad=True, dtype=torch.float64)

    # Solve the Poisson equation N times

    #print(poisson.v)

    #f * poisson.v * dx

    u = poisson(f, g)
    print(u)

    # Construct functional
    J = u.sum()

    # Execute the backward pass
    J.backward()

    # Extract gradients
    dJdf = f.grad
    dJdg = g.grad
