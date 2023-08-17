# Elements not requiring stabilization
from dolfin import *
from xii import block_form, ii_assemble, apply_bc
import xii.meshing.cell_centers as centers
from xii import *
import utils

Tangent = lambda v, n: v - n*dot(v, n)    
Normal = lambda v, n: n*dot(v, n)

nitsche_penalty = lambda u: Constant(10)

# Work horses --------------------------------------------------------
def get_system_base(Velm, Qelm, boundaries, normals, *, mms_data, bcs, K):
    '''Things without bcs (we apply transformations later)'''
    mesh = boundaries.mesh()
    Velm, Qelm = Velm(mesh.ufl_cell()), Qelm(mesh.ufl_cell())

    V = FunctionSpace(mesh, Velm)
    Q = FunctionSpace(mesh, Qelm)    
    K = Constant(K)
    
    Ltags = ()
    if 'L' in bcs.values():
        Ltags = tuple(tag for (tag, bc) in bcs.items() if bc == 'L')
        bmesh = EmbeddedMesh(boundaries, Ltags)
        
        assert Velm.degree() == 1 and Velm.family() == 'Raviart-Thomas'
        # Multiplier space
        L = FunctionSpace(bmesh, 'DG', 0)
        W = [V, Q, L]
        u, p, l = map(TrialFunction, W)
        v, q, dl = map(TestFunction, W)        
    else:
        W = [V, Q]
        u, p = map(TrialFunction, W)
        v, q = map(TestFunction, W)
    
    a = block_form(W, 2)
    a[0][0] = (1/K)*inner(u, v)*dx
    a[0][1] = -inner(p, div(v))*dx
    a[1][0] = -inner(q, div(u))*dx

    L = block_form(W, 1)
    L[1] = -inner(mms_data['f'], q)*dx

    # The multiplier part
    if Ltags:
        dx_ = Measure('dx', domain=bmesh)
        n_ = OuterNormal(bmesh, [0, 0])
        Tu, Tv = Trace(u, bmesh), Trace(v, bmesh)

        a[0][2] = inner(dot(Tv, n_), l)*dx_
        a[2][0] = inner(dot(Tu, n_), dl)*dx_
        
        L[2] = inner(dot(mms_data['u'], n_), dl)*dx_

    return a, L, W


def assemble_system_with_strong_bcs(a, L, W, bcs, boundaries, normals, *, mms_data, K):
    '''Apply Dirichlet bcs to the system'''
    V, Q = W[:2]
    u, v = TrialFunction(V), TestFunction(V)
    
    mesh = boundaries.mesh()
    assert mesh.geometry().dim() == 2
    # Let's get stuff ready for setting boundary conditions
    dB = Measure('ds', domain=mesh, subdomain_data=boundaries)    
    p0 = mms_data['p']
    
    V_bcs = []
    has_LM = False
    for tag, bc_type in bcs.items():
        n = normals[tag]
        if bc_type == 'F':
            V_bcs.append(DirichletBC(V, mms_data['u'], boundaries, tag))
            
        elif bc_type == 'P':
            L[0] -= inner(p0, dot(v, n))*dB(tag)

        elif bc_type == 'L':
            has_LM = True
            continue
        else:
            assert False

    if not has_LM:
        W_bcs = [V_bcs, []]
    else:
        W_bcs = [V_bcs, [], []]
    
    A, b = map(ii_assemble, (a, L))
    A, b = apply_bc(A, b, bcs=W_bcs)

    return A, b, W, W_bcs

# Now it should be easy peasy ----------------------------------------
def get_system_dirichlet(Velm, Qelm, boundaries, normals, *, mms_data, bcs, K):
    '''Symmetric formulation ith strongly enforced bcs'''
    a, L, W = get_system_base(Velm, Qelm, boundaries, normals=normals,
                              mms_data=mms_data, bcs=bcs, K=K)
    return assemble_system_with_strong_bcs(a, L, W,
                                           bcs=bcs, boundaries=boundaries, normals=normals,
                                           mms_data=mms_data, K=K)
