import numpy as np
from autograd import grad
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS
from scipy.optimize import SR1

def f1(x):
    return x[0]
def f2(x):
    return x[1]
def g(x):
    return -(x[0]**2 + x[1]**2 -0.81)
def h(x):
    return -x[0] - 0.9

g_dx = grad(g)
cons = []
cons.append({'type': 'ineq',
          'fun' : lambda x: g(x),
          'jac' : lambda x: g_dx(x)})
A = np.array([[-1, -1]])
b = np.array([[1.0]])
for i in range(0,1):
    cons.append({'type': 'ineq', 'fun' : lambda x: np.dot(A[i, ], x) - b[i], 'jac' : lambda x: A[i, ]})
cons = tuple(cons)
bounds = Bounds([-1, -1], [1, 1])
class CFG:
    epsilon = 1e-2
    hatd = [1.111,1]

def find_min(f,n,cons):
    x = np.random.rand(1,n).tolist()[0]
    res = minimize(f, x,  jac="2-point",hess=BFGS(),
                constraints=cons,method='trust-constr', options={'disp': False},bounds=bounds)
    return res.x
x1 = find_min(f1,2,cons)
x2 = find_min(f2,2,cons)
m = [f1(x1),f2(x2)]
M = [max(f1(x1),f1(x2)), max(f2(x1),f2(x2))]
print(f"m: {m}, M: {M} ")
cons1 = []
cons1.append({'type': 'ineq',
          'fun' : lambda x: (f1(x)-M[0]),
          'jac' : lambda x: grad(f1)(x)})
cons1.append({'type': 'ineq',
          'fun' : lambda x: (f2(x)-M[1]),
          'jac' : lambda x: grad(f2)(x)})
cons1.append({'type': 'ineq',
          'fun' : lambda x: g(x),
          'jac' : lambda x: grad(g)(x)})
A = np.array([[-1, -1]])
b = np.array([1.0])
# for i in range(0,1):
#     cons1.append({'type': 'ineq', 'fun' : lambda x: np.dot(A[i, ], x) - b[i], 'jac' : lambda x: A[i,]})
cons1 = tuple(cons1)
def phi_monotonic(h,n):
    x = np.random.rand(1,n).tolist()[0]
    #x = [-1,]
    res = minimize(h, x, jac="2-point",hess=BFGS(),
                constraints=cons1,method='trust-constr', options={'disp': False})
    return res.x, h(res.x)
# def t(x,f1,f2,v):
#     return max((f1(x)-v[0])/CFG.hatd[0], (f2(x)-v[1])/CFG.hatd[1])
def t(x):
    return x[2]
# cons2 = []
# cons2.append({'type': 'ineq',
#           'fun' : lambda x: g(x),
#           'jac' : lambda x: grad(g)(x)})
# A = np.array([[-1, -1]])
# b = np.array([1.0])
# for i in range(0,1):
#     cons2.append({'type': 'ineq', 'fun' : lambda x: np.dot(A[i, ], x) - b[i], 'jac' : lambda x: A[i,]})
# cons2 = tuple(cons2)
def f1_new(x,v):
    #print(x[0] - v[0] - x[2]*CFG.hatd[0])
    return x[0] - v[0] - x[2]*np.array(CFG.hatd)[0]
def f2_new(x,v):
    return x[1]- v[1] - x[2]*np.array(CFG.hatd)[1]
def pv_solve(t,f1_new,f2_new,v,n):
    
    #x = [-1,]
    cons2 = []
    cons2.append({'type': 'ineq',
            'fun' : lambda x: g(x),
            'jac' : lambda x: grad(g)(x)})
    cons2.append({'type': 'ineq',
            'fun' : lambda x: f1_new(x,v) ,
            'jac' : lambda x: grad(f1_new)(x,v)})
    cons2.append({'type': 'ineq',
            'fun' : lambda x: f2_new(x,v),
            'jac' : lambda x: grad(f2_new)(x,v)})
    bounds1 = Bounds([-1, -1,-1], [1, 1,1])
    A = np.array([[-1, -1,0]])
    b = np.array([1.0])
    for i in range(0,1):
        cons2.append({'type': 'ineq', 'fun' : lambda x: np.dot(A[i, ], x) - b[i], 'jac' : lambda x: A[i,]})
    cons2 = tuple(cons2)
    x = np.random.rand(1,n).tolist()[0]
    res = minimize(t, x,jac="2-point",hess=BFGS(),
                constraints=cons2,method='trust-constr', options={'disp': False},bounds=bounds1)
    return res.x
V = M
x_phi, phiVal = phi_monotonic(h,2)
print(f"x_phi: {x_phi}, phiVal: {phiVal} ")
def ParTask_new(V):
    V_loc = []
    #for i in range(1):
    v = V
    x = pv_solve(t,f1_new,f2_new,v,3)
    # print(t(x,f1_new,f2_new,v),CFG.hatd)
    #print(t(x),CFG.hatd)
    # w = v + t(x,f1,f2,v)*np.array(CFG.hatd)
    w = v + t(x)*np.array(CFG.hatd)
    y = [f1(x), f2(x)]
    alpha_loc = h(x)
    x_sol_loc = x
    y_sol_loc = y
    return x_sol_loc, y_sol_loc, alpha_loc
x_sol_loc, y_sol_loc, alpha_loc = ParTask_new(V)
print(f"x_sol_loc: {x_sol_loc}, y_sol_loc: {y_sol_loc}, alpha_loc: {alpha_loc} ")
    #V_loc = [V_loc;GetNewProperElements(v,w,p,m,M)]
# for i in range(100):
#     [All_V_loc, All_alpha_loc, All_x_sol_loc, All_y_sol_loc] = ParTask_new(h,f,A,b,g,V(1,:),hatd,p,m,M)


