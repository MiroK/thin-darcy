from gmshnics.interopt import msh_gmsh_model, mesh_from_gmsh
import dolfin as df
import numpy as np
import gmsh


def squares(length, radius, width, ncells, nrefs, generator=None):
    '''Foo'''
    if radius == np.inf:
        if generator == 'gmsh':
            yield from flat_channel_gmsh_meshes(length, width, ncells, nrefs)
        else:
            yield from flat_channel_meshes(length, ncells, nrefs)
    else:
        yield from curved_channel_meshes(length, radius, width, ncells, nrefs)


def flat_channel_meshes(L, ncells, nrefs):
    '''(0, L) x (0, 1)'''
    assert L >= 1

    bdries = {1: df.CompiledSubDomain('near(x[0], -L/2)', L=L),
              2: df.CompiledSubDomain('near(x[0], L/2)', L=L),
              3: df.CompiledSubDomain('near(x[1], -0.5)'),
              4: df.CompiledSubDomain('near(x[1], 0.5)')}

    for k in range(nrefs):
        ncells_y = ncells*2**k
        ncells_x = int(L)*ncells_y        
        mesh = df.RectangleMesh(df.Point(-L/2, -0.5), df.Point(L/2, 0.5), ncells_x, ncells_y)

        boundary = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
        [bdry.mark(boundary, tag) for tag, bdry in bdries.items()]

        assert set(np.unique(boundary.array())) == {0, 1, 2, 3, 4}

        xmin, ymin = mesh.coordinates().min(axis=0)
        xmax, ymax = mesh.coordinates().max(axis=0)

        assert abs(xmin+L/2) < 1E-10 and abs(ymin+0.5) < 1E-10
        assert abs(xmax-L/2) < 1E-10 and abs(ymax-0.5) < 1E-10

        normals = {1: df.Constant((-1, 0)),
                   2: df.Constant((1, 0)),
                   3: df.Constant((0, -1)),
                   4: df.Constant((0, 1))}
                            
        yield boundary, normals

        
def curved_channel(model, length, radius, width=1):
    # D-----------------C                
    # |                 |
    # A-----------------B
    #
    #         O
    fac = model.occ
    origin = fac.addPoint(0, 0, z=0)

    # The arc angle is determined by archlength in the channel midheight
    # which we want to be length
    theta = length/2/radius

    assert theta < np.pi/2
    
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    # AB sit on a smaller arc
    r0 = np.array([0, radius-width/2])
    A = R@r0
    B = R.T@r0

    A = fac.addPoint(A[0], A[1], 0, width)
    B = fac.addPoint(B[0], B[1], 0, width)
    lower_arc = fac.addCircleArc(A, origin, B)
    
    # Now the top
    r1 = np.array([0, radius+width/2])    
    D = R@r1
    C = R.T@r1

    D = fac.addPoint(D[0], D[1], 0, width)
    C = fac.addPoint(C[0], C[1], 0, width)
    upper_arc = fac.addCircleArc(C, origin, D)

    # Sides
    left = fac.addLine(D, A)
    right = fac.addLine(B, C)

    loop = fac.addCurveLoop([lower_arc, right, upper_arc, left])
    channel = fac.addPlaneSurface([loop])
    fac.synchronize()

    model.addPhysicalGroup(2, [channel], 1)
    model.addPhysicalGroup(1, [lower_arc], 3)
    model.addPhysicalGroup(1, [upper_arc], 4)
    model.addPhysicalGroup(1, [left], 1)
    model.addPhysicalGroup(1, [right], 2)    
    fac.synchronize()

    dydx1 = model.getValue(0, D, [])[:2] - model.getValue(0, A, [])[:2]
    dydx1 = dydx1/np.linalg.norm(dydx1)

    dydx2 = model.getValue(0, B, [])[:2] - model.getValue(0, C, [])[:2]
    dydx2 = dydx2/np.linalg.norm(dydx2)    

    normals = {1: df.Constant((-dydx1[1], dydx1[0])),
               2: df.Constant((-dydx2[1], dydx2[0])),
               3: df.Expression(('-(x[0]-x0)/r', '-(x[1]-x1)/r'), x0=0, x1=0,
                                r=radius-width/2, degree=1),
               4: df.Expression(('(x[0]-x0)/r', '(x[1]-x1)/r'), x0=0, x1=0,
                                r=radius+width/2, degree=1)}
                               
    return model, normals


def curved_channel_meshes(length, radius, width, ncells, nrefs):
    '''Generate refinments'''
    gmsh.initialize()
    model = gmsh.model
    model, normals = curved_channel(model, length, radius, width=1)

    scale = 1./ncells
    for k in range(nrefs):
        gmsh.option.setNumber('Mesh.MeshSizeMax', scale/2**(k+1))
        
        nodes, topologies = msh_gmsh_model(model, 2)
        mesh, entity_functions = mesh_from_gmsh(nodes, topologies)

        yield entity_functions[1], normals
        
        # Ready for next round
        gmsh.model.mesh.clear()
    # At this point we are done with gmsh
    gmsh.finalize()


def flat_channel_gmsh(model, length, width=1):
    # D-----------------C                
    # |                 |
    # A-----------------B
    #
    #         O
    fac = model.occ

    A = fac.addPoint(-length/2, -0.5, 0)
    B = fac.addPoint(length/2, -0.5, 0)
    C = fac.addPoint(length/2, 0.5, 0)
    D = fac.addPoint(-length/2, 0.5, 0)    

    # Sides
    upper_arc = fac.addLine(C, D)
    lower_arc = fac.addLine(A, B)
    left = fac.addLine(D, A)
    right = fac.addLine(B, C)

    loop = fac.addCurveLoop([lower_arc, right, upper_arc, left])
    channel = fac.addPlaneSurface([loop])
    fac.synchronize()

    model.addPhysicalGroup(2, [channel], 1)
    model.addPhysicalGroup(1, [lower_arc], 3)
    model.addPhysicalGroup(1, [upper_arc], 4)
    model.addPhysicalGroup(1, [left], 1)
    model.addPhysicalGroup(1, [right], 2)    
    fac.synchronize()

    normals = {1: df.Constant((-1, 0)),
               2: df.Constant((1, 0)),
               3: df.Constant((0, -1)),
               4: df.Constant((0, 1))}    
                               
    return model, normals


def flat_channel_gmsh_meshes(length, width, ncells, nrefs):
    '''Generate refinments'''
    gmsh.initialize()
    model = gmsh.model
    model, normals = flat_channel_gmsh(model, length, width=1)

    left, = model.getEntitiesForPhysicalGroup(1, 1)
    right, = model.getEntitiesForPhysicalGroup(1, 2)    
    bottom, = model.getEntitiesForPhysicalGroup(1, 3)
    top, = model.getEntitiesForPhysicalGroup(1, 4)
    
    scale = 1./ncells
    for k in range(nrefs):
        # gmsh.option.setNumber('Mesh.MeshSizeFactor', scale/2**(k))

        ncells_ = 2**k*ncells
        model.mesh.setTransfiniteCurve(left, ncells_)
        model.mesh.setTransfiniteCurve(right, ncells_)    
        model.mesh.setTransfiniteCurve(bottom, int(length*ncells_))
        model.mesh.setTransfiniteCurve(top, int(length*ncells_))
        
        nodes, topologies = msh_gmsh_model(model, 2)
        mesh, entity_functions = mesh_from_gmsh(nodes, topologies)

        yield entity_functions[1], normals
        
        # Ready for next round
        gmsh.model.mesh.clear()
    # At this point we are done with gmsh
    gmsh.finalize()
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    import dolfin as df
    
    for ans in squares(length=1, radius=4, width=1, ncells=2, nrefs=3):
        print(ans.mesh().hmin())
    df.File('foo.pvd') << ans
