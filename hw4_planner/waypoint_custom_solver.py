
"""
landing trajectory with minimum acceleration and obstacle avoidance
        2/17/22 - RWB
"""
import numpy as np
from scipy.linalg import norm
from scipy.optimize import minimize

def fin_step(f, x, args):
    J = np.zeros_like(x)
    eps = 1e-10
    f0 = f(x, *args)
    if  isinstance(f0, np.ndarray):
        J = np.zeros((len(f0), len(x)))
    
    else:
        J = np.zeros((1, len(x)))

    for idx, val in enumerate(x):
        delta = np.copy(x)
        delta[idx] += eps
        f_delta = f(delta, *args)
        J[:,idx] = (f_delta - f0)/eps
    
    return J

def SQP_QN(x0, fun, cons, tau_opt=1e-3, tau_feas=1e-3, args=None):
    x = x0.reshape((-1,))
    xPrev = x
    f = fun(x, *args)
    h = cons(x, *args)
    grad = fin_step(fun, x, args).reshape(-1,1)
    gradPrev = grad
    Jh = fin_step(cons, x, args)
    JhPrev = Jh

    p = np.zeros((len(x)+len(h), 1))

    lmda = np.zeros((Jh.shape[0],1))
    alpha=1

    JL = grad + Jh.T@lmda

    k = 0

    while (norm(JL, np.inf) > tau_opt or norm(h, np.inf) > tau_feas) and k < 30000:
        print( norm(JL, np.inf), norm(h, np.inf), k )
        # Quasi Newton Hessian
        if k == 0:
            H = np.eye(len(x))
        else:
            s = (x - xPrev).reshape((-1,1))
            y= JL - (gradPrev + JhPrev.T@lmda)
            H -= H@s@s.T@H / (s.T@H@s) + y@y.T / (y.T@s)

        # Solve approx QP problem
        A = np.block([[H, Jh.T],
                      [Jh, np.zeros((Jh.shape[0], Jh.shape[0]))]])
        b = np.concatenate((-JL.flatten(), -h))
        p = np.linalg.solve(A,b)
        # p = QP(p, H, JL, Jh, h)  

        px = p[:len(x)]
        pl = p[len(x):].reshape((-1,1))

        lmda += pl

        alpha = .01 # TODO linesearch
        xPrev = np.copy(x)
        x += alpha * px

        gradPrev = np.copy(grad)
        JhPrev = np.copy(Jh)
        grad = fin_step(fun, x, args).reshape((-1,1))
        Jh = fin_step(cons, x, args)
        h = cons(x, *args)
        # print(h)
        JL = grad + Jh.T@lmda

        k+=1

    return x


def QP_fun(p, H, JL):
    p = p.reshape((-1,1))
    return 0.5 * p.T@H@p + JL.T@p

def QP_cons(p, h, Jh):
    return Jh@p + h

def QP(p, H, JL, Jh, h):
    cons_eq = [{'type':'eq', 'fun':QP_cons, 'args': (h, Jh)}]
    res = minimize(QP_fun, p, args=(H, JL), method='SLSQP', cons=cons_eq)
    return res.x
    





        