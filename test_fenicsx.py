from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
import ufl


domain = mesh.create_unit_square(MPI.COMM_WORLD,10,10)
x = ufl.SpatialCoordinate(domain)


V = fem.functionspace(domain, ('P', 1))



tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

bc_exp = fem.Expression(fem.Constant(domain,0.0),V.element.interpolation_points())


uD = fem.Function(V)
uD.interpolate(bc_exp)

bc = fem.dirichletbc(uD, boundary_dofs)

u = ufl.TestFunction(V)
v = ufl.TrialFunction(V)

f1 = fem.Constant(domain,0.0)
f2 = fem.Constant(domain,0.0)

a = ufl.inner(ufl.grad(u),ufl.grad(v)) * ufl.dx
L1 = f1 * v * ufl.dx
L2 = f2 * v * ufl.dx

#u1 = fem.Function(V)
#solve(a == L1, u1, bc)
P1 = LinearProblem(
        a,
        L1,
        bcs=[bc],
        petsc_options={
            "ksp_type":"preonly",
            "pc_type":"lu"
            },
        #petsc_options_prefix="Poisson"
        )
u1 = P1.solve()
#u2 = fem.Function(V)

P2 = LinearProblem(
        a,
        L2,
        bcs=[bc],
        petsc_options={
            "ksp_type":"preonly",
            "pc_type":"lu"
            },
        #petsc_options_prefix="Poisson"
        )
u2 = P2.solve()