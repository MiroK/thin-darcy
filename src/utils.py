from dolfin import dot, Constant, VectorElement, FiniteElement
from block import block_mat, block_vec
from collections import namedtuple
from functools import partial
import dolfin as df
import numpy as np


def tangent(v, n):
    '''Tangent part of vector v on surface with normal n'''
    return v - dot(v, n)*n

# We assume tagging as
#   4
# 1   2      
#   3
# Which vector components are normal to the boundary?
COMPONENT_INDICES_2D = {1: 0, 2: 0, 3: 1, 4: 1}
# What is the normal of the surface?
NORMAL_VECTORS_2D = {1: Constant((-1, 0)),
                     2: Constant((1, 0)),
                     3: Constant((0, -1)),
                     4: Constant((0, 1))}

# TODO: later extend this to 3d
component_indices = lambda mesh: {2: COMPONENT_INDICES_2D}[mesh.geometry().dim()]

normal_vectors = lambda mesh: {2: NORMAL_VECTORS_2D}[mesh.geometry().dim()]


# Some pretty printing
GREEN = '\033[1;37;32m%s\033[0m'
RED = '\033[1;37;31m%s\033[0m'
BLUE = '\033[1;37;34m%s\033[0m'
YELLOW = '\033[1;37;33m%s\033[0m'
MAGENTA = '\033[1;37;35m%s\033[0m'
CYAN = '\033[1;37;36m%s\033[0m'

def print_color(color, string):
    '''Print with color'''
    print(color % string)
    # NOTE: this is here just to have something to test
    return color

print_red = partial(print_color, RED)
print_green = partial(print_color, GREEN)
print_blue = partial(print_color, BLUE)    
print_yellow = partial(print_color, YELLOW)
print_magenta = partial(print_color, MAGENTA)
print_cyan = partial(print_color, CYAN)    


def parse_elements(Velm, Qelm):
    '''Recipes how to make the Darcy elements'''
    Vfamily, Vdeg = Velm.upper().split('_')
    Vdeg = int(Vdeg)
    assert Vdeg > 0

    Qfamily, Qdeg = Qelm.upper().split('_')
    Qdeg = int(Qdeg)
    assert Qdeg >= 0

    # Special handle mini
    Velm = {'CG': lambda cell, deg=Vdeg: VectorElement('Lagrange', cell, deg),
            'CR': lambda cell, deg=Vdeg: VectorElement('Crouzeix-Raviart', cell, deg),
            'RT': lambda cell, deg=Vdeg: FiniteElement('Raviart-Thomas', cell, deg),
            'BDM': lambda cell, deg=Vdeg: FiniteElement('Brezzi-Douglas-Marini', cell, deg)
            }[Vfamily]

    Qelm = {'CG': lambda cell, deg=Qdeg: FiniteElement('Lagrange', cell, deg),
            'DG': lambda cell, deg=Qdeg: FiniteElement('Discontinuous Lagrange', cell, deg)
            }[Qfamily]

    return Velm, Qelm


RESULT_CONTAINERS = {'sanity': namedtuple('Sanity', ('h', 'ndofs', 'eu', 'reu', 'ep', 'rep', 'qual')),
                     'iters': namedtuple('Iters', ('h', 'ndofs', 'eu', 'reu', 'ep', 'rep', 'niters', 'cond', 'qual', 'bep0', 'beph')),
                     'eigs': namedtuple('Eigs', ('h', 'ndofs', 'lmin', 'lmax', 'cond', 'qual')),
                     'ieigs': namedtuple('Eigs', ('h', 'ndofs', 'lmin', 'lmax', 'cond', 'qual'))}
                     
                     
def parse_bcs(bcs):
    '''4letter string to dict'''
    assert len(bcs) == 4
    assert set(bcs) <= set(('F', 'P', 'L'))

    bcs = dict(enumerate(bcs, 1))    

    return bcs


def handle_pressure_nullspace(A, b, W, Wbcs, nullspace):
    '''Add cstr (p, 1)*dx'''
    # Let's find pressure
    k, found = 0, False
    while not found:
        Vk = W[k].ufl_element()
        cell = Vk.cell()
        # Scalar space not ont en embedded surface
        if Vk.value_shape() == () and cell.geometric_dimension() == cell.topological_dimension():
            found = True
        else:
            k += 1
    assert found
    Q = W[k]
    
    mesh = Q.mesh()
    R = df.VectorFunctionSpace(mesh, 'R', 0, len(nullspace))
    # New space
    WxR = W + [R]

    rs, drs = df.TrialFunction(R), df.TestFunction(R)
    p, q = df.TrialFunction(Q), df.TestFunction(Q)

    Ablocks = A.blocks.tolist()
    # Now we want to add to the matrix ...
    # First into a column
    for row in range(len(W)):
        if row != k:
            Ablocks[row].append(0)
        else:
            # The constraint
            Ablocks[row].append(df.assemble(
                sum(df.inner(zi*rsi, q)*df.dx for zi, rsi in zip(nullspace, rs))
            ))
    # Now last row
    C = df.assemble(sum(df.inner(zj*drsj, p)*df.dx for zj, drsj in zip(nullspace, drs)))
    D = df.assemble(df.Constant(0)*df.inner(rs, drs)*df.dx)
    Ablocks.append([C if col == k else 0
                    for col in range(len(WxR)-1)] + [D])
    # ... and the vector ...
    bblocks = b.blocks.tolist()
    bblocks.append(df.assemble(df.inner(df.Constant((0, )*len(nullspace)), drs)*df.dx))

    A, b = block_mat(Ablocks), block_vec(bblocks)

    # ... and the bcs
    Wbcs.append([])
    
    return A, b, WxR, Wbcs


def handle_rigid_motions_nullspace(A, b, W, Wbcs, nullspace):
    '''Add cstr (p, 1)*dx'''
    # Let's find pressure
    V = W[0]
    
    mesh = V.mesh()
    R = df.VectorFunctionSpace(mesh, 'R', 0, len(nullspace))
    # New space
    WxR = W + [R]

    r, dr = df.TrialFunction(R), df.TestFunction(R)
    u, v = df.TrialFunction(V), df.TestFunction(V)

    Ablocks = A.blocks.tolist()
    # Now we want to add to the matrix ...
    k = 0
    # First into a column
    for row in range(len(W)):
        if row != k:
            Ablocks[row].append(0)
        else:
            # The constraint
            Ablocks[row].append(df.assemble(sum(df.inner(vec*ri, v)*df.dx for ri, vec in zip(r, nullspace))))
    # Now last row
    C = df.assemble(sum(df.inner(vec*dri, u)*df.dx for dri, vec in zip(dr, nullspace)))
    D = df.assemble(df.Constant(0)*df.inner(r, dr)*df.dx)
    Ablocks.append([C if col == k else 0
                    for col in range(len(WxR)-1)] + [D])
    # ... and the vector ...
    bblocks = b.blocks.tolist()
    bblocks.append(df.assemble(df.inner(df.Constant((0, )*len(nullspace)), dr)*df.dx))

    A, b = block_mat(Ablocks), block_vec(bblocks)

    # ... and the bcs
    Wbcs.append([])
    
    return A, b, WxR, Wbcs


def errornorm(u, uh, which, degree_rise):
    '''`Dolfin's erronorm` with handling of L20'''
    if which != 'L20':
        return df.errornorm(u, uh, which, degree_rise)
    # Normalize to mean 0
    assert uh.ufl_shape == ()

    def normalize(fh):
        mesh = fh.function_space().mesh()
        mean = df.assemble(fh*df.dx)/df.assemble(df.Constant(1)*df.dx(domain=mesh))
        fh.vector().axpy(-mean,
                         df.interpolate(df.Constant(1), fh.function_space()).vector())
        return fh

    Vh = uh.function_space()
    uh0 = df.Function(Vh)
    uh0.vector().axpy(1, uh.vector())
    uh0 = normalize(uh0)
    
    Eh = df.FunctionSpace(Vh.mesh(), 'Discontinuous Lagrange', Vh.ufl_element().degree()+degree_rise)
    u0 = df.interpolate(u, Eh)
    u0 = normalize(u0)
    
    return errornorm(u0, uh0, 'L2', degree_rise)


def is_rectangle(mesh, tol_=1E-10):
    '''More like a sanity check'''
    # 3---2
    # |   |
    # 0---1
    assert mesh.geometry().dim() == 2
    
    x0, y0 = mesh.coordinates().min(axis=0)
    x2, y2 = mesh.coordinates().max(axis=0)

    dx = x2 - x0
    dy = y2 - y0

    area0 = dx*dy
    area = sum(c.volume() for c in df.cells(mesh))
    h = mesh.hmin()
    
    return abs(area - area0) < h*tol_


def is_flat_surface(boundaries, tag, dG, tol_=1E-10):
    '''Assuming simply connected'''
    mesh = boundaries.mesh()
    assert mesh.geometry().dim() == 2
    assert mesh.topology().dim() == 2

    _, f2v = mesh.init(1, 0), mesh.topology()(1, 0)
    vidx = np.unique(np.hstack([f.entities(0) for f in df.SubsetIterator(boundaries, tag)]))
    x = mesh.coordinates()[vidx]
    x0 = x.min(axis=0)
    x1 = x.max(axis=0)

    area = np.linalg.norm(x1-x0)
    area0 = sum(df.Edge(mesh, f.index()).length() for f in df.SubsetIterator(boundaries, tag))

    return abs(area - area0) < tol_


def rm_basis(mesh):
    '''3 functions(Expressions) that are the rigid motions of the body.'''
    assert mesh.geometry().dim() == 2
    
    r = df.SpatialCoordinate(mesh)
    dX = df.dx(domain=mesh)
    volume = df.assemble(df.Constant(1)*dX)

    translations = [df.Expression(('u', 'v'), degree=0, u=v[0]/df.sqrt(volume), v=v[1]/df.sqrt(volume))
                    for v in np.eye(2)]    
    # Center of mass
    c = np.array([df.assemble(xi*dX) for xi in r])
    c /= volume

    x0, y0 = mesh.coordinates().min(axis=0)
    x1, y1 = mesh.coordinates().max(axis=0)

    points = (c, (0.5*(x0+x1), y0), (0.5*(x0+x1), y1), (x0, 0.5*(y0+y1)), (x1, 0.5*(y0+y1))) 
    x, y = df.SpatialCoordinate(mesh)

    rotations = []
    for center in points:
        rot = df.as_vector((-(y-df.Constant(center[1])), (x-df.Constant(center[0]))))
    
        n = df.assemble(df.inner(rot, rot)*dX)
        rot = df.Expression(('-(x[1]-c1)/A', '(x[0]-c0)/A'), degree=1, c0=center[0], c1=center[1], A=df.sqrt(n))
        rotations.append(rot)

    Z = translations + rotations

    return Z


def has_rm_nullspace(mat, V, bcs, tol_=1E-10):
    '''Check'''
    candidates = rm_basis(V.mesh())
    nullspace = []
    for vec in candidates:
        x = df.interpolate(vec, V).vector()
        values = x.get_local()
        # Does it satisfy the boundary conditions
        if any(np.linalg.norm(values[list(bc.get_boundary_values().keys())]) > 0
               for bc in bcs):
            continue
        y = mat*x
        y.norm('l2') < tol_ and nullspace.append(vec)
    return nullspace


def pressure_basis(mesh):
    '''Scaled 1'''
    vol = df.assemble(df.Constant(1)*df.dx(domain=mesh))
    return [df.Constant(1/df.sqrt(vol))]
    

def has_pressure_nullspace(A, W, tol_=1E-10):
    '''Check'''
    Q = W[1]
    
    candidates = pressure_basis(Q.mesh())
    nullspace = []
    while candidates:
        vec = candidates.pop()
        
        x = df.interpolate(vec, Q).vector()
        if len(W) == 2:
            bx = block_vec([0, x])
        else:
            one = df.interpolate(df.Constant(1), W[-1]).vector()
            bx = block_vec([0, x, one])
        
        y = A*bx
        norm = sum(yi.norm('l2') for yi in y if yi)
        norm < tol_ and nullspace.append(vec)
    return nullspace


def randomize(vec):
    x = vec.get_local()
    x[:] = np.random.rand(len(x))
    vec.set_local(x)

    return vec

# --------------------------------------------------------------------

if __name__ == '__main__':
    mesh = df.UnitSquareMesh(32, 32)

    facet_f = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    df.CompiledSubDomain('near(x[0], 0)').mark(facet_f, 1)
    dB = df.Measure('ds', domain=mesh, subdomain_data=facet_f)

    mesh.rotate(45)
    
    foo = is_flat_surface(facet_f, tag=1, dG=dB)
    print(foo)
    df.File('foo.pvd') << mesh
