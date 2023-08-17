from dolfin import *
import sympy as sp
import utils
import ulfy
# Manufactured solution - always the same function
# We have axis aligned rectangle domain in mind with edges marked by
#   4
# 1   2
#   3
#

def darcy_2d(K_value=1, l_value=1):
    x, y = SpatialCoordinate(UnitSquareMesh(1, 1))
    K0, length0 = Constant(K_value), Constant(l_value)

    p = sin(pi*(x-y))
    u = -K0*grad(p)
    f = div(u)

    K_, l_ = sp.symbols('K l')
    subs = {K0: K_, length0: l_}
    as_expression = lambda v: ulfy.Expression(v, degree=6, subs=subs, K=K_value, l=l_value)
    
    components, normals = utils.COMPONENT_INDICES_2D, utils.NORMAL_VECTORS_2D

    return {'f': as_expression(f),
            'u': as_expression(u),
            'p': as_expression(p)}
    
