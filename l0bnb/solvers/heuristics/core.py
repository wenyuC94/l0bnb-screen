import copy
from time import time
from collections import namedtuple
import math

import numpy as np
from numba.typed import List
from numba import njit

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


from ..utils import set2arr
from ._L0L2heuristics import L0L2_ASCD_solve
from ..mosek import relax_mosek


EPSILON = np.finfo('float').eps

def _initialize_algo(X, y, S_diag, beta, z, solver, upper_support_heuristic, p):
    assert upper_support_heuristic in {"all", "nonzeros", "rounding","support"}
    if beta is None and z is None:
        upper_support = set(range(p))
    elif z is not None:
        if upper_support_heuristic == 'all':
            upper_support = set(range(p))
        elif upper_support_heuristic == 'rounding':
            upper_support = set(np.where(np.round(z))[0])
        elif upper_support_heuristic == 'nonzeros':
            upper_support = set(np.where(z)[0])
        elif upper_support_heuristic == 'support':
            upper_support = set(np.where(beta)[0])
    else:
        if upper_support_heuristic == 'all':
            upper_support = set(range(p))
        else:
            upper_support = set(np.where(beta)[0])
    
    upper_active_set = set2arr(upper_support)
    upper_active_set_mask = np.full(p, False)
    upper_active_set_mask[upper_active_set] = True
    
    if 'AS' not in solver:
        if beta is not None:
            beta[~upper_active_set_mask] = 0
            Xb = X@beta
        else:
            beta, Xb = trivial_soln(X)
    elif beta is not None:
        Xb = X@beta
    else:
        Xb = None
    if beta is not None:
        support = set(np.where(beta)[0])
    else:
        support = None
    return beta, Xb, support, upper_support, upper_active_set, upper_active_set_mask

def heuristic_solve(loss_name, X, y, l0, l2, M, delta=1., solver="L0L2_ASCD", upper_support_heuristic="all", beta=None, z=None, \
                    S_diag=None, rel_tol=1e-4, timing=False, maxtime=np.inf, verbose=False, **kwargs):
    
    assert solver in {'L0L2_ASCD','Mosek'}
    if S_diag is None:
        S_diag = np.linalg.norm(X,axis=0)**2
    
    beta, Xb, support, upper_support, upper_active_set, upper_active_set_mask = _initialize_algo(X, y, S_diag, beta, z, solver, upper_support_heuristic, X.shape[1])
    if solver == 'L0L2_ASCD':
        if beta is not None:
            warm_start = {'beta':beta,'support':support}
        else:
            warm_start = None
        cd_tol = kwargs.get('cd_tol', 1e-4)
        cd_max_itr = kwargs.get('cd_max_itr', 100)
        kkt_max_itr = kwargs.get('kkt_max_itr', 100)
        beta, cost, Xb, active_set = L0L2_ASCD_solve(loss_name, X, y, l0, l2, M, upper_active_set_mask, delta, S_diag, warm_start, \
                    cd_tol, cd_max_itr, rel_tol, kkt_max_itr, timing, maxtime, verbose)
        support = set(active_set)
    elif solver == 'Mosek':
        z_round = np.zeros(p)
        z_round[list(upper_support)] = 1
        beta,z,cost,dual_cost= relax_mosek(loss_name, X, y, l0, l2, M,\
                                                  zlb=z_round,zub=z_round,loss_params=None)
        support = set(np.where(beta)[0])
        Xb = X@beta
    return beta, cost, Xb, support
    

# def heuristic_solve(Y, l0, l2, M, solver="L0L2_CDPSI", support_type="all", beta=None, z=None, \
#                     S_diag=None, rel_tol=1e-4, cd_max_itr=100, verbose=False, **kwargs):
#     p = Y.shape[1]
#     assert solver in {"L0L2_CDPSI", "L0L2_CD", "L2_CDApprox", "L2_CD", "L0L2_ASCD", "L0L2_ASCDPSI"}
#     if S_diag is None:
#         S_diag = np.linalg.norm(Y,axis=0)**2
    
#     beta, Xb, support, active_set = _initialize_algo(Y, S_diag, beta, z, solver, support_type, p)
#     if solver in {"L0L2_CDPSI", "L0L2_CD"}:
#         cost = get_L0L2_cost(Y, beta, Xb, l0, l2, M, active_set)
#     elif solver in {"L2_CD","L2_CDApprox"}:
#         cost = get_L2_primal_cost(Y, beta, Xb, l2, M, active_set)
    
#     if solver == "L0L2_CD":
#         beta, cost, Xb = L0L2_CD(Y, beta, cost, l0, l2, M, S_diag, active_set, Xb, rel_tol, cd_max_itr, verbose)
#     elif solver == "L0L2_CDPSI":
#         swap_max_itr = kwargs.get("swap_max_itr", 10)
#         beta, cost, Xb = L0L2_CDPSI(Y, beta, cost, l0, l2, M, S_diag, active_set, Xb, rel_tol, cd_max_itr, swap_max_itr, verbose)
#     elif solver == "L2_CDApprox":
#         beta, cost, Xb = L2_CD(Y, beta, cost, l2, M, S_diag, active_set, Xb, rel_tol, maxiter, verbose)
#         cost += len(support)*l0
#     elif solver == "L2_CD":
#         kkt_max_itr = kwargs.get("kkt_max_itr",100)
#         warm_start = dict()
#         warm_start['beta'] = beta
#         warm_start['Xb'] = Xb
#         beta, cost, Xb = L2_CD_solve(Y, l2, M, support, S_diag, warm_start, rel_tol,cd_max_itr,kkt_max_itr,verbose)
#         cost += len(support)*l0
#     elif solver in {"L0L2_ASCD", "L0L2_ASCDPSI"}:
#         if beta is not None:
#             warm_start = dict()
#             warm_start['beta'] = beta
#             warm_start['Xb'] = Xb
#             warm_start['support'] = set([(i,j) for i,j in np.argwhere(beta) if i < j])
#         else:
#             warm_start = None
#         if solver == "L0L2_ASCD":
#             kkt_max_itr = kwargs.get("kkt_max_itr",100)
#             beta, cost, Xb, support = L0L2_ASCD(Y, l0, l2, M, support, S_diag, warm_start, rel_tol, cd_max_itr, rel_tol, kkt_max_itr, verbose=verbose)
#         elif solver == "L0L2_ASCDPSI":
#             kkt_max_itr = kwargs.get("kkt_max_itr",100)
#             swap_max_itr = kwargs.get("swap_max_itr", 10)
#             beta, cost, Xb, support = L0L2_ASCDPSI(Y, l0, l2, M, support, S_diag, warm_start, rel_tol, cd_max_itr, swap_max_itr, rel_tol, kkt_max_itr, verbose=verbose)
#     support = set([(i,j) for [i,j] in np.argwhere(beta) if i<j])
#     return beta, cost, Xb, support

# class L0L2Solver:
#     def __init__(self, X, assumed_centered=False, cholesky=False):
#         self.n, self.p, self.X, self.X_mean, self.Y, self.S_diag = preprocess(X,assume_centered,cholesky)
        
#     def solve(self, l0, l2, M, solver="ASCD", warm_start=None, verbose=False):
#         pass