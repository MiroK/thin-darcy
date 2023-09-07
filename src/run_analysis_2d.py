import argparse, tabulate, importlib
import numpy as np
import os
# Our stuff
from block.algebraic.petsc.solver import petsc_py_wrapper
from petsc4py import PETSc
from block.algebraic.petsc import KSP
from block.iterative import MinRes
from xii import ii_Function, ii_convert, ReductionOperator
import dolfin as df

import geometry, utils, preconditioning, slepc_utils, mms


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('analysis', type=str, choices=('sanity', 'iters', 'eigs', 'ieigs'))
# Geometry
parser.add_argument('-length', type=int, default=1, help='L in (0, L) x (0, 1) which is the domain shape')
# NOTE: If the domain is curved we do not experct errors from solvers to converge for some
# bc types as e.g. setting components of the velocity requires boundaries to be aligned with
# coordinate axis
parser.add_argument('-radius', type=float, default=np.inf, help='Radius of the channel if curved')
parser.add_argument('-ncells0', type=int, default=2, help='Initial number of cells')
parser.add_argument('-nrefs', type=int, default=1, help='Number of refinements')
parser.add_argument('-meshgen', type=str, default='native', help='Pick native meshes or Gmsh')
# Problem
parser.add_argument('-K', type=float, default=1.0, help='Permeability value')
parser.add_argument('-mms_kind', type=str, default='basic', help='Select manufactured solution')
# Discretization
parser.add_argument('-Velm', type=str, default='RT_1', help='Velocity element')
parser.add_argument('-Qelm', type=str, default='DG_0', help='Pressure element')
# Boundary conditions
parser.add_argument('-bcs', type=str, default='PPFF', help='Type of bcs on the 4 edges')
# How are we solving the system
parser.add_argument('-ksp_type', type=str, default='minres')
parser.add_argument('-pc_type', type=str, default='rieszk', help='Which preconditioner')
#
# IO
parser.add_argument('-error_cvrg', type=int, default=1, help='Look at error convergence')
parser.add_argument('-save', type=int, default=0, choices=(0, 1), help='Save graphics')    

args, _ = parser.parse_known_args()
# Don't recompile when value changes
K = df.Constant(args.K)

# Some things for book keeping
result_dir = f'./results/{args.analysis}/{args.Velm}-{args.Qelm}/radius{args.radius}'
not os.path.exists(result_dir) and os.makedirs(result_dir)

def get_path(what, ext, args=vars(args), ignore_keys=('analysis', 'save', 'ncells0', 'nrefs', 'error_cvrg')):
    template_path = '_'.join([what] + [f'{key.upper()}{value}' for key, value in args.items()
                                       if key not in ignore_keys])
    template_path = '.'.join([template_path, ext])
    return os.path.join(result_dir, template_path)

# What should we store
result_row = utils.RESULT_CONTAINERS[args.analysis]

results = []
with open(get_path('results', 'txt'), 'w') as out:
    out.write('# %s\n' % ' '.join(result_row._fields))                

# Decide discretization
Velm, Qelm = utils.parse_elements(args.Velm, args.Qelm)

# Decide arrangement of boundary conditions
bcs = utils.parse_bcs(args.bcs)

# In periodic case, it better be sane and on a rectangle
# Pick the manufactured solution
mms_data = mms.darcy_2d(K_value=args.K, l_value=args.length)

# Decide system: this has two parts: do we stabilize ... ?
from discretization import get_system_dirichlet as get_system

seed = PETSc.Random().create(comm=PETSc.COMM_WORLD)

meshes = geometry.squares(length=args.length, radius=args.radius, width=1, ncells=args.ncells0, nrefs=args.nrefs,
                          generator=args.meshgen)

spectrum_history = []
h0, eu0, ep0 = None, None, None
for boundary, normals in meshes:
    mesh = boundary.mesh()
    
    A, b, W, Wbcs = get_system(Velm=Velm,
                               Qelm=Qelm,
                               boundaries=boundary,
                               normals=normals,
                               mms_data=mms_data,
                               bcs=bcs,
                               K=K)

    pressure_nullspace = utils.has_pressure_nullspace(A, W)
    if 'L' in args.bcs:
        assert not pressure_nullspace
    
    # Add LM for (p, 1) if needed
    if pressure_nullspace:
        # Check basis
        gram = np.array([
            df.assemble(df.inner(ui, vj)*df.dx(domain=mesh))
            for ui in pressure_nullspace
            for vj in pressure_nullspace
        ]).reshape(*(len(pressure_nullspace), )*2)
        assert np.linalg.norm(gram - np.eye(len(pressure_nullspace))) < 1E-10
        
        A, b, W, Wbcs = utils.handle_pressure_nullspace(A, b, W, Wbcs, pressure_nullspace)
        nullspace = pressure_nullspace
    else:
        nullspace = []
        
    row = {field: -1 for field in result_row._fields}
    # Common to all
    row['ndofs'] = sum(Wi.dim() for Wi in W)
    row['h'] = mesh.hmin()
    row['qual'] = df.MeshQuality.radius_ratio_min_max(mesh)[0]
    # Just look at manufactured solution; nothing much else to compute
    if args.analysis == 'sanity':
        wh = ii_Function(W)
        A, b = map(ii_convert, (A, b))

        solver = df.LUSolver('umfpack')
        solver.solve(A, wh.vector(), b)
        
    # Look at the iteration count
    elif args.analysis == 'iters':
        B = preconditioning.form_riesz_map(A, W, Wbcs, bcs, K=K, which=args.pc_type,
                                           nullspace=nullspace)

        # FIXME: random initial guess
        wh = ii_Function(W)
        x = wh.block_vec()

        if args.ksp_type == 'minres':
            Ainv = KSP(A, precond=B, 
                       # PETScOptions
                       ksp_type='minres',
                       ksp_rtol=1E-12,
                       ksp_view=None,
                       ksp_max_it=300,
                       ksp_monitor_true_residual=None,
                       ksp_initial_guess_nonzero=1,
                       ksp_converged_reason=None)
        elif args.ksp_type == 'gmres':
            Ainv = KSP(A, precond=B,
                       # PETScOptions
                       ksp_type='gmres',
                       ksp_rtol=1E-12,
                       ksp_view=None,
                       ksp_initial_guess_nonzero=1,                       
                       ksp_monitor_true_residual=None,
                       ksp_converged_reason=None)
        else:
            cbk = lambda k, x, r, b=b, A=A: print(f'\titer {k} |r|={r}')                    
            Ainv = MinRes(A, precond=B, initial_guess=x,
                          tolerance=1E-12, callback=cbk, maxiter=500)

        x = Ainv*b
        [whi.vector().axpy(1, xi) for whi, xi in zip(wh, x)]

        row['niters'] = len(Ainv.residuals)

        cond = -1
        try:
            eigw = Ainv.eigenvalue_estimates()
        except RuntimeError:
            eigw = []
        if len(eigw):
            cond = max(np.abs(eigw))/min(np.abs(eigw))
        row['cond'] = cond
    # Look at the condition number
    elif args.analysis == 'eigs':
        # Pass RM
        B = preconditioning.form_inner_product(A, W, Wbcs, bcs, K=K, which=args.pc_type,
                                               nullspace=nullspace)
        A_, B_ = (df.as_backend_type(ii_convert(X)).mat() for X in (A, B))
        
        (lmin, vmin), (lmax, vmax) = slepc_utils.min_max_eigenvalues(A_, B_, W, nev=1, Z=None)
        cond = abs(lmax)/abs(lmin)

        row['lmin'], row['lmax'], row['cond'] = (lmin, lmax, cond)

        wh = vmin
    else:
        raise ValueError
        
    # Are we computing the right thing?
    if args.error_cvrg:
        uh, ph = wh[0], wh[1]
        h = uh.function_space().mesh().hmin()
        # Swich RM
        eu = utils.errornorm(mms_data['u'], uh, 'Hdiv0', degree_rise=1)
        
        ep = utils.errornorm(mms_data['p'], ph,
                             'L20' if len(nullspace) else 'L2',
                             degree_rise=1)

        from hsmg.hseig import HsEig
        if 'L' in args.bcs:
            bph = wh[2]
            bep0 = utils.errornorm(mms_data['p'], bph, 'L2', degree_rise=2)

            L = W[2]
            Hs = HsEig(L, s=0.5).collapse()
            eh = df.interpolate(mms_data['p'], L).vector()
            eh.axpy(-1, bph.vector())
            beph = df.sqrt(eh.inner(Hs*eh))
        else:
            ds_ = df.ds(domain=mesh, subdomain_data=boundary, subdomain_id=2, metadata={'quadrateture_degree': 4})
            bep0 = df.assemble(df.inner(mms_data['p'] - ph,
                                        mms_data['p'] - ph)*ds_)
            bep0 = df.sqrt(bep0)

            from xii import EmbeddedMesh
            
            L = df.FunctionSpace(EmbeddedMesh(boundary, 2), 'DG', 0)
            Hs = HsEig(L, s=0.5).collapse()
            eh = df.interpolate(mms_data['p'], L).vector()
            eh.axpy(-1, df.interpolate(ph, L).vector())
            beph = df.sqrt(eh.inner(Hs*eh))
            
        row['eu'], row['ep'], row['bep0'], row['beph'] = eu, ep, bep0, beph

        rate_u, rate_p = -1, -1
        if eu0 is not None:
            rate_u = df.ln(eu/eu0)/df.ln(h/h0)
            rate_p = df.ln(ep/ep0)/df.ln(h/h0)
            row['reu'], row['rep'] = (f'{val:.2f}' for val in (rate_u, rate_p))
        h0, eu0, ep0 = h, eu, ep

    row = tuple(row[val] for val in result_row._fields)
    results.append(row)
    # Print it and store it
    utils.print_green(tabulate.tabulate(results, headers=result_row._fields))
    with open(get_path('results', 'txt'), 'a') as out:
        out.write('# %s\n' % (' '.join(tuple(map(str, row)))))

utils.print_blue(f'{W[0].ufl_element()} x {W[1].ufl_element()}')

if args.save:
    df.File(get_path('uh', 'pvd')) << uh
    df.File(get_path('ph', 'pvd')) << ph

    if ph.function_space().ufl_element().degree() == 0:
        df.File(get_path('ph_CG1', 'pvd')) << df.interpolate(ph,
                                                             df.FunctionSpace(ph.function_space().mesh(), 'CG', 1))
    # Just for info
    utils.print_red(os.path.abspath(get_path('uh', 'pvd')))
    utils.print_red(os.path.abspath(get_path('ph', 'pvd')))

    u_true = df.interpolate(mms_data['u'], uh.function_space())
    p_true = df.interpolate(mms_data['p'], ph.function_space())
    df.File(get_path('u_true', 'pvd')) << u_true
    df.File(get_path('p_true', 'pvd')) << p_true
    
    # Get the error too
    uh.vector().axpy(-1, u_true.vector())
    ph.vector().axpy(-1, p_true.vector())    
    df.File(get_path('euh', 'pvd')) << uh
    df.File(get_path('eph', 'pvd')) << ph
