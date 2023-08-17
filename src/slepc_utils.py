from petsc4py import PETSc
from slepc4py import SLEPc
import dolfin as df
import numpy as np
import xii


def extreme_eigenvalues(A, B, V, which, nev=1, Z=None, shift=0, eps_problem_type=None):
    ''''Solve Ax = l B*x for the extreme eigenvalue'''
    opts = PETSc.Options()
    dopts = opts.getAll()
    

    'eps_max_it' not in dopts and opts.setValue('eps_max_it', 50_000)
    'eps_nev' not in dopts and opts.setValue('eps_nev', nev)
    'eps_monitor' not in dopts and opts.setValue('eps_monitor', None)
    'eps_view' not in dopts and opts.setValue('eps_view', None)

    'eps_type' not in dopts and opts.setValue('eps_type', 'krylovschur')
    opts.setValue('st_ksp_rtol', 1E-10)
    #opts.setValue('st_ksp_monitor_true_residual', None)

    shift and opts.setValue('st_shift', shift)

    # Setup the eigensolver
    E = SLEPc.EPS().create()
    Z is not None and (E.setDeflationSpace(Z), print(f'|Deflation space|={len(Z)}'))

    if B is not None:
        E.setProblemType(SLEPc.EPS.ProblemType.GHEP if eps_problem_type is None else eps_problem_type)
        opts.setValue('eps_tol', 1E-4)
        # NOTE: this might be too low to avoid numerical modes in some cases
        # Taylor-Hood RRDD for example
    else:
        E.setProblemType(SLEPc.EPS.ProblemType.NHEP if eps_problem_type is None else eps_problem_type)
        opts.setValue('eps_tol', 1E-10)
    E.setOperators(A, B)

    E.setWhichEigenpairs(which)

    if isinstance(V, (list, tuple)):
        gdim = V[0].mesh().geometry().dim()
    else:
        gdim = V.mesh().geometry().dim()
    # Some sizes that can be handled reasonbly well by a direct solver
    ksp_iter_size = 1E6 if gdim == 2 else 1E4
    
    if A.size[0] > ksp_iter_size:
        opts.setValue('st_ksp_type', 'cg')
        opts.setValue('st_pc_type', 'hypre')
    E.setFromOptions()
    
    E.solve()

    its = E.getIterationNumber()
    nconv = E.getConverged()

    pairs = []
    for i in range(nconv):
        mode = A.createVecLeft()
        val = E.getEigenpair(i, mode)

        Amode = mode.copy()
        A.mult(mode, Amode)

        Bmode = mode.copy()
        if B is not None:
            B.mult(mode, Bmode)
        
        Amode.axpy(-abs(val), Bmode)
        error = Amode.norm(2)
        print(f'\tPair {i} with eigenvalue {val} has error {error}')

        if isinstance(V, (tuple, list)):
            foo_mode = xii.ii_Function(V)
            first = 0
            for i, foo_mode_i in enumerate(foo_mode):
                ndofs = V[i].dim()
                
                last = first + ndofs
                foo_mode_i.vector().set_local(mode.array_r[np.arange(first, last)])
                first = last
        else:
            foo_mode = df.Function(V)
            foo_mode.vector()[:] = df.PETScVector(mode)
        
        pairs.append((val, foo_mode))
        
    return E, pairs, its


def min_max_eigenvalues(A, B, V, nev=1, Z=None):
    '''Eigenvalues for condition number computations'''
    if B is not None:
        eps_problem_type = SLEPc.EPS.ProblemType.GHEP
    else:
        eps_problem_type = SLEPc.EPS.ProblemType.NHEP
    
    # Largest 
    E, eigws_max, _ = extreme_eigenvalues(A, B, V, Z=None, nev=1,
                                          which=SLEPc.EPS.Which.LARGEST_MAGNITUDE,
                                          eps_problem_type=eps_problem_type)
    # Sort starting from larges
    eigws_max = sorted(eigws_max, key=lambda p: abs(p[0]), reverse=True)

    pairs = iter(eigws_max)
    # Get first
    eigw_max, mode_max = next(pairs)
    assert np.imag(eigw_max) < 10*E.getTolerances()[0], np.imag(eigw_max)
    eigw_max = np.real(eigw_max)

    # Smallest
    E, eigws_min, _ = extreme_eigenvalues(A, B, V, Z=None, nev=1,
                                          which=SLEPc.EPS.Which.SMALLEST_MAGNITUDE,
                                          eps_problem_type=eps_problem_type)
    # Sort starting from smallest
    eigws_min = sorted(eigws_min, key=lambda p: abs(p[0]), reverse=False)

    pairs = iter(eigws_min)
    # Get first
    eigw_min, mode_min = next(pairs)
    assert np.imag(eigw_min) < 10*E.getTolerances()[0], np.imag(eigw_min)
    eigw_min = np.real(eigw_min)

    return (eigw_min, mode_min), (eigw_max, mode_max)


def spectrum(A, B, Z=None, eps_problem_type=None):
    ''''Solve Ax = l B*x for the extreme eigenvalue'''
    # Setup the eigensolver
    E = SLEPc.EPS().create()
    Z is not None and (E.setDeflationSpace(Z), print(f'|Deflation space|={len(Z)}'))

    E.setProblemType(SLEPc.EPS.ProblemType.GHEP if eps_problem_type is None else eps_problem_type)
    E.setType(SLEPc.EPS.Type.LAPACK)
    E.setWhichEigenpairs(SLEPc.EPS.Which.ALL)
    
    E.setOperators(A, B)
    E.setUp()
    
    E.solve()

    its = E.getIterationNumber()
    nconv = E.getConverged()
    eigenvalues = [E.getEigenvalue(i) for i in range(nconv)]

    return eigenvalues
