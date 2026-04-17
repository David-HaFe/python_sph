

from petsc4py import PETSc
from mpi4py import MPI
import ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    create_vector,
    set_bc,
)
import numpy
import numpy.typing as npt

### problem specific parameters ###
t = 0.0  # Start time
T = 3.0  # End time
num_steps = 20  # Number of time steps
dt = (T - t) / num_steps  # Time step size
alpha = 3.0
beta = 1.2

### define mesh and appropriate function space ###
nx, ny = 5, 5
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

### class that represents exact solution ###
class ExactSolution:
    def __init__(self, alpha: float, beta: float, t: float):
        self.alpha = alpha
        self.beta = beta
        self.t = t

    def __call__(self, x: npt.NDArray[numpy.floating]) -> npt.NDArray[numpy.floating]:
        return 1 + x[0] ** 2 + self.alpha * x[1] ** 2 + self.beta * self.t

u_exact = ExactSolution(alpha, beta, t)

### dirichlet boundary condition ###
u_D = fem.Function(V)
u_D.interpolate(u_exact)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets))

### variational formulation ###
u_n = fem.Function(V)
u_n.interpolate(u_exact)

f = fem.Constant(domain, beta - 2 - 2 * alpha)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
F = (
    u * v * ufl.dx
    + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    - (u_n + dt * f) * v * ufl.dx
)
a = fem.form(ufl.lhs(F))
L = fem.form(ufl.rhs(F))

### matrix and vector for linear problem ###
A = assemble_matrix(a, bcs=[bc])
A.assemble()
# b = create_vector(fem.extract_function_spaces(L))
b = create_vector(L) # <- DAVID modification
uh = fem.Function(V)

### linear variational solver ###
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
pc = solver.getPC()
pc.setType(PETSc.PC.Type.LU)

### solving time dependent problem ###
for n in range(num_steps):
    # Update Diriclet boundary condition
    u_exact.t += dt
    u_D.interpolate(u_exact)

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, L)

    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [a], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    # Solve linear problem
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array

A.destroy()
b.destroy()
solver.destroy()

### verifying solution ###
# Compute L2 error and error at nodes
V_ex = fem.functionspace(domain, ("Lagrange", 2))
u_ex = fem.Function(V_ex)
u_ex.interpolate(u_exact)
error_L2 = numpy.sqrt(
    domain.comm.allreduce(
        fem.assemble_scalar(fem.form((uh - u_ex) ** 2 * ufl.dx)), op=MPI.SUM
    )
)
if domain.comm.rank == 0:
    print(f"L2-error: {error_L2:.2e}")

# Compute values at mesh vertices
error_max = domain.comm.allreduce(
    numpy.max(numpy.abs(uh.x.array - u_D.x.array)), op=MPI.MAX
)
if domain.comm.rank == 0:
    print(f"Error_max: {error_max:.2e}")


