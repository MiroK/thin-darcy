from block.algebraic.petsc import LU, AMG
from utils import rm_basis, has_rm_nullspace
from block import block_mat
from petsc4py import PETSc
import scipy.sparse as sp
import dolfin as df
from dolfin import *
import xii

from hsmg.hseig import HsEig

# ----

def form_riesz_map(A, W, Wbcs, bcs, K, which, nullspace):
    
    if which == 'rieszk':
        if 'L' in bcs.values():
            assert not nullspace
            return form_riesz_map_rieszk_lm(A, W, Wbcs, bcs, K, nullspace)
        return form_riesz_map_rieszk(A, W, Wbcs, bcs, K, nullspace)

    elif which == 'rieszknobc':
        assert 'L' in bcs.values()
        assert not nullspace

        return form_riesz_map_rieszk_lm(A, W, Wbcs, bcs, K, nullspace, force_no_bcs=True)
    
    elif which == 'riesznok':
        return form_riesz_map_rieszk(A, W, Wbcs, bcs, Constant(1), nullspace)
    else:
        raise ValueError


def form_riesz_map_rieszk(A, W, Wbcs, bcs, K, nullspace):
    '''Standard Stokes preconditioner for 2x2'''
    V, Q = W[:2]

    shape = None
    if nullspace:
        shape, = set(v.ufl_shape for v in nullspace)

    u, v = TrialFunction(V), TestFunction(V)
    a_prec0 = (1/K)*inner(u, v)*dx + (1/K)*inner(div(u), div(v))*dx
    L_prec0 = inner(Constant((0, )*len(u)), v)*dx
    B0, _ = assemble_system(a_prec0, L_prec0, Wbcs[0])
    B0 = LU(B0)
    
    p, q = df.TrialFunction(Q), df.TestFunction(Q)
    # B1 = df.assemble(K*df.inner(p, q)*df.dx)
    # # Add
    # if not isinstance(A[1][1], (int, float)):
    #     B1 = xii.ii_convert(B1 - A[1][1])
    p, q = df.TrialFunction(Q), df.TestFunction(Q)
    
    B1_M = df.assemble(K*df.inner(p, q)*df.dx)

    boundaries, tag = Wbcs[0][0].domain_args
    Qbc_tags = set((1, 2, 3, 4)) - set((Vbc.domain_args[1] for Vbc in Wbcs[0]))
    print('>>>', Qbc_tags, '<<<')
    Qbcs = [df.DirichletBC(Q, df.Constant(0), boundaries, tag) for tag in Qbc_tags]
    B1_A, _ = df.assemble_system(df.inner(df.grad(p), df.grad(q))*df.dx,
                                 df.inner(df.Constant(0), q)*df.dx,
                                 Qbcs)
    # Add
    if not isinstance(A[1][1], (int, float)):
        B1_M = xii.ii_convert(B1_M - A[1][1])
    B1 = LU(B1_M) + LU(B1_A) if inv == 'lu' else AMG(B1_M) + AMG(B1_A)
        
    if len(W) == 2:
        return xii.block_diag_mat([B0, B1])

    # ---
    R = W[-1]
    rs, drs = df.TrialFunction(R), df.TestFunction(R)
    B2 = df.assemble(sum((1/K)*df.inner(zi*ri, zj*drj)*df.dx
                         for zi, ri in zip(nullspace, rs)
                         for zj, drj in zip(nullspace, drs)))
    
    return xii.block_diag_mat([LU(B0), LU(B1), LU(B2)])    


def form_riesz_map_rieszk_lm(A, W, Wbcs, bcs, K, nullspace, force_no_bcs=False):
    '''Standard Stokes preconditioner for 2x2'''
    V, Q, L = W

    shape = None
    if nullspace:
        shape, = set(v.ufl_shape for v in nullspace)

    u, v = TrialFunction(V), TestFunction(V)
    a_prec0 = (1/K)*inner(u, v)*dx + (1/K)*inner(div(u), div(v))*dx
    L_prec0 = inner(Constant((0, )*len(u)), v)*dx
    B0, _ = assemble_system(a_prec0, L_prec0, Wbcs[0])
    
    p, q = df.TrialFunction(Q), df.TestFunction(Q)
    B1 = df.assemble(K*df.inner(p, q)*df.dx)
    # Add
    if not isinstance(A[1][1], (int, float)):
        B1 = xii.ii_convert(B1 - A[1][1])

    bcs_str = ''.join([bcs[k] for k in (1, 2, 3, 4)])

    assert bcs_str.endswith('LPP') or bcs_str.endswith('LFF')

    if bcs_str.endswith('LPP') and not force_no_bcs:
        Lfacet_f = MeshFunction('size_t', L.mesh(), L.mesh().topology().dim()-1, 0)
        DomainBoundary().mark(Lfacet_f, 1)
        Lbcs = [(Lfacet_f, 1)]
    else:
        Lbcs = None
        
    B2 = HsEig(L, s=0.5, kappa=K, bcs=Lbcs)
    B2 = B2.collapse()

    return xii.block_diag_mat([LU(B0), LU(B1), LU(B2)])


#-- Eigenvalues

def form_inner_product(A, W, Wbcs, bcs, K, which, nullspace):
    
    if which == 'rieszk':
        if 'L' in bcs.values():
            assert not nullspace
            return form_inner_product_rieszk_lm(A, W, Wbcs, bcs, K, nullspace)
        return form_inner_product_rieszk(A, W, Wbcs, bcs, K, nullspace)
    elif which == 'riesznok':
        return form_inner_product_rieszk(A, W, Wbcs, bcs, Constant(1), nullspace)

    elif which == 'rieszknobc':
        assert 'L' in bcs.values()
        assert not nullspace

        return form_inner_product_rieszk_lm(A, W, Wbcs, bcs, K, nullspace, force_no_bcs=True)
    
    else:
        raise ValueError

# ----

def form_inner_product_rieszk(A, W, Wbcs, bcs, K, nullspace):
    '''Standard Stokes preconditioner for 2x2'''
    assert len(W) in (2, 3)
    
    V, Q = W[:2]

    shape = None
    if nullspace:
        shape, = set(v.ufl_shape for v in nullspace)

    u, v = TrialFunction(V), TestFunction(V)
    a_prec0 = (1/K)*inner(u, v)*dx + (1/K)*inner(div(u), div(v))*dx
    L_prec0 = inner(Constant((0, )*len(u)), v)*dx
    B0, _ = assemble_system(a_prec0, L_prec0, Wbcs[0])
    
    p, q = df.TrialFunction(Q), df.TestFunction(Q)
    B1 = df.assemble(K*df.inner(p, q)*df.dx)
    # Add
    if not isinstance(A[1][1], (int, float)):
        B1 = xii.ii_convert(B1 - A[1][1])

    if len(W) == 2:
        return xii.block_diag_mat([B0, B1])

    # ---
    R = W[-1]
    rs, drs = df.TrialFunction(R), df.TestFunction(R)
    B2 = df.assemble(sum((1/K)*df.inner(zi*ri, zj*drj)*df.dx
                         for zi, ri in zip(nullspace, rs)
                         for zj, drj in zip(nullspace, drs)))

    return xii.block_diag_mat([B0, B1, B2])    


def form_inner_product_rieszk_lm(A, W, Wbcs, bcs, K, nullspace, force_no_bcs=False):
    '''Standard Stokes preconditioner for 2x2'''
    assert len(W) in (2, 3)
    
    V, Q, L = W

    shape = None
    if nullspace:
        shape, = set(v.ufl_shape for v in nullspace)

    u, v = TrialFunction(V), TestFunction(V)
    a_prec0 = (1/K)*inner(u, v)*dx + (1/K)*inner(div(u), div(v))*dx
    L_prec0 = inner(Constant((0, )*len(u)), v)*dx
    B0, _ = assemble_system(a_prec0, L_prec0, Wbcs[0])
    
    p, q = df.TrialFunction(Q), df.TestFunction(Q)
    B1 = df.assemble(K*df.inner(p, q)*df.dx)
    # Add
    if not isinstance(A[1][1], (int, float)):
        B1 = xii.ii_convert(B1 - A[1][1])

    bcs_str = ''.join([bcs[k] for k in (1, 2, 3, 4)])

    assert bcs_str.endswith('LPP') or bcs_str.endswith('LFF')

    if bcs_str.endswith('LPP') and not force_no_bcs:
        Lfacet_f = MeshFunction('size_t', L.mesh(), L.mesh().topology().dim()-1, 0)
        DomainBoundary().mark(Lfacet_f, 1)
        Lbcs = [(Lfacet_f, 1)]
    else:
        Lbcs = None
        
    B2 = HsEig(L, s=0.5, kappa=K, bcs=Lbcs)
    B2 = B2.collapse()

    return xii.block_diag_mat([B0, B1, B2])    
