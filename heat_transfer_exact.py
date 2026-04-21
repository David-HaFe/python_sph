from petsc4py import PETSc
from mpi4py import MPI
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    create_vector,
    set_bc,
)
import numpy as np
import numpy.typing as npt
import csv
from config import (
    no_particles_x,
    no_particles_y,
    border,
    t0,
    t1,
    dt,
    no_steps,
)

### problem specific parameters ###
alpha = 3.0
beta = 1.2

### define mesh and appropriate function space ###
nx, ny = no_particles_x - 1, no_particles_y - 1
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([-border, -border]), np.array([border, border])],
    [nx, ny],
    mesh.CellType.triangle,
)  # <- DAVID modification
# domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))


### class that represents exact solution ###
class ExactSolution:
    def __init__(self, alpha: float, beta: float, t: float):
        self.alpha = alpha
        self.beta = beta
        self.t = t

    def __call__(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
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
b = create_vector(L)  # <- DAVID modification
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
error_L2 = np.sqrt(
    domain.comm.allreduce(
        fem.assemble_scalar(fem.form((uh - u_ex) ** 2 * ufl.dx)), op=MPI.SUM
    )
)
if domain.comm.rank == 0:
    print(f"L2-error: {error_L2:.2e}")

# Compute values at mesh vertices
error_max = domain.comm.allreduce(np.max(np.abs(uh.x.array - u_D.x.array)), op=MPI.MAX)
if domain.comm.rank == 0:
    print(f"Error_max: {error_max:.2e}")

### export to csv ###
filename = "csvs/exact_solution.csv"

# Build 300 evenly spaced evaluation points over the mesh extent
coords = V_ex.tabulate_dof_coordinates()
x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
x_eval = np.linspace(x_min, x_max, N_POINTS).reshape(-1, 1)
# Pad to 3D as required by FEniCSx eval()
x_eval_3d = np.hstack([x_eval, np.zeros((N_POINTS, 2))])

# Write header before the time loop
if domain.comm.rank == 0:
    with open(filename, "w+", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "data"])

# --- Inside your time loop ---
t_start, t_end, dt = 0.0, 1.0, 0.01
t_values = np.arange(t_start, t_end, dt)
for t in t_values:
    # ... solve / update u_ex here ...

    # Evaluate u_ex at the 300 points
    cells = []
    points_on_proc = []
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, x_eval_3d)
    colliding_cells = geometry.compute_colliding_cells(
        domain, cell_candidates, x_eval_3d
    )
    for i, pt in enumerate(x_eval_3d):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(pt)
            cells.append(colliding_cells.links(i)[0])

    u_values = u_ex.eval(np.array(points_on_proc), cells)

    # Gather across MPI ranks
    all_values = domain.comm.gather(u_values.flatten(), root=0)

    if domain.comm.rank == 0:
        full_array = np.concatenate(all_values)
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [t, np.array2string(full_array, separator=" ", max_line_width=np.inf)]
            )
