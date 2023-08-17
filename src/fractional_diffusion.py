from dolfin import *
from hsmg.hseig import Hs0Eig, HsEig
import numpy as np

import gmshnics as gs

mesh, _ = gs.gCircle(center=[0, 0], radius=1, size=0.05)

mesh = BoundaryMesh(mesh, 'exterior')
V = FunctionSpace(mesh, 'DG', 0)
u, v = TrialFunction(V), TestFunction(V)
M = assemble(inner(u, v)*dx)

facet_f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
DomainBoundary().mark(facet_f, 1)
bcs = [(facet_f, 0.5)]

s = 1.0
foo = 'sine'

# -------------------

Hs = HsEig(V, s=s, bcs=None).collapse()

x_dofs = V.tabulate_dof_coordinates()

import cmath
theta = np.array(list(map(cmath.phase, x_dofs[:, 0] + 1j*x_dofs[:, 1])))

idx = np.argsort(theta)

values = {'sine': np.sin,
          'cosine': np.cos,
          'other': lambda arg: np.abs(np.sin(arg)) #np.where(arg > 0, np.cos(np.sin(2*arg)),
                                        #np.abs(np.sin(2*arg)))
          }[foo](theta)  # np.cos(np.pi*np.sin(theta))
f = Function(V)
f.vector().set_local(values)

x = f.vector()

table = [values[idx]]
for k in range(1):
    y = Hs*x
    solve(M, x, y)
    table.append(x.get_local()[idx])

import matplotlib.pyplot as plt

smooth = Function(V)
smooth.vector()[:] = x




x_dofs = theta[idx]

plt.figure()
for y_dofs in table:
    plt.plot(x_dofs, y_dofs)#/np.max(y_dofs))
plt.show()

data = np.c_[x_dofs, np.array(table).T]
header = ' '.join(['x'] + [f'D{k}' for k in range(len(table))])
np.savetxt(f'eval_{foo}_{s}.txt', data, header=header)
