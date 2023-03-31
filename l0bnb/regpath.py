from copy import deepcopy
import numpy as np
from time import time
from collections import namedtuple

from .data_utils import preprocess
from .loss_utils import compute_grad_Lipschitz_sigma, compute_grad
from .solvers.utils import trivial_soln, get_active_set_mask, set2arr
from .solvers.heuristics import heuristic_solve
from .core import BNBTree

def compute_next_lambda(loss_name,X,y,l2,M,S_diag,beta,Xb,support,delta=1.):
    p = X.shape[1]
    if len(support) == p:
        return 0.
    sigma = compute_grad_Lipschitz_sigma(loss_name)
    grad = compute_grad(loss_name, y, X, beta, Xb, delta)
    a = sigma*S_diag/2
    b = -a*2*beta+grad
    criterion = np.where(np.abs(b)/2/(a+l2)<=M, b**2/4/(a+l2), -a*M**2+np.abs(b)*M-l2*M**2)
    active_set_mask = get_active_set_mask(set2arr(support), p)
    return np.max(criterion[~active_set_mask])


def fit_path_L0L2(loss_name, X, y, delta=1.,lambda2 = 0.01, M = np.inf, solver='ASCD',
                  lambda0_grid = None, maxSuppSize = None, n_lambda0 = 100, scaleDownFactor = 0.8, 
                  rel_tol=1e-6, cd_max_itr=100, kkt_max_itr=100, cd_tol=1e-4, verbose=True):
    assert solver in {"ASCD"}
    n,p,X,y,S_diag = preprocess(X, y)

    if maxSuppSize is None:
        maxSuppSize = p
    _sol_str = 'beta sol_time support cost'
    Solution = namedtuple('Solution', _sol_str)
    sols = [] 
    terminate = False
    iteration_num = 0
    if verbose:
        print("L0L2 Heuristics Started.")
    if lambda0_grid is not None:
        lambda0_grid = sorted(lambda0_grid, reverse = True)
        beta = None
        while not terminate:
            l0 = lambda0_grid[iteration_num]
            st  = time()
            if verbose:
                print(l0,lambda2)
            beta, cost, Xb, support = heuristic_solve(loss_name, X, y, l0, lambda2, M, delta=delta, solver=f"L0L2_{solver}", upper_support_heuristic="all", beta=beta, z=None, \
                    S_diag=S_diag, rel_tol=rel_tol, timing=False, maxtime=np.inf, verbose=False, kkt_max_itr=kkt_max_itr, cd_max_itr=cd_max_itr, cd_tol=cd_tol)
            sols.append(Solution(beta=np.copy(beta), sol_time = time()-st, support = support, cost = cost))
            if verbose:
                print("Iteration: " + str(iteration_num) + ". Number of non-zeros: ",len(support))
            iteration_num += 1
            if iteration_num == len(lambda0_grid):
                terminate = True
            if len(support) >= maxSuppSize:
                terminate = True
                lambda0_grid = lambda0_grid[:iteration_num]
    else:
        beta, Xb = trivial_soln(X)
        support = set()
        lambda0_grid = []
        eps_factor = min(1e-3, (1-scaleDownFactor)*1e-2)
        while not terminate:
            if iteration_num == 0:
                l0 = compute_next_lambda(loss_name,X,y,lambda2,M,S_diag,beta,Xb,support,delta)*(1-eps_factor)
            else:
                l0 = min(l0*scaleDownFactor, compute_next_lambda(loss_name,X,y,lambda2,M,S_diag,beta,Xb,support,delta)*(1-eps_factor))
            lambda0_grid.append(l0)
            st  = time()
            if verbose:
                print(l0,lambda2)
            beta, cost, Xb, support = heuristic_solve(loss_name, X, y, l0, lambda2, M, delta=delta, solver=f"L0L2_{solver}", upper_support_heuristic="all", beta=beta, z=None, \
                    S_diag=S_diag, rel_tol=rel_tol, timing=False, maxtime=np.inf, verbose=False, kkt_max_itr=kkt_max_itr, cd_max_itr=cd_max_itr, cd_tol=cd_tol)
            sols.append(Solution(beta=np.copy(beta), sol_time = time()-st, support = support, cost = cost))
            if verbose:
                print("Iteration: " + str(iteration_num) + ". Number of non-zeros: ",len(support))
            iteration_num += 1
            if iteration_num == n_lambda0:
                terminate = True
            if len(support) >= maxSuppSize:
                terminate = True
            if l0 == 0:
                terminate = True
            
    return lambda0_grid, sols


def fit_path_BnB(loss_name, X, y, delta=1.,lambda2 = 0.01, M = np.inf, M_multiplier=1.05,
                  lambda0_grid = None, maxSuppSize = None, n_lambda0 = 100, scaleDownFactor = 0.8, 
                  rel_tol=1e-6, cd_max_itr=100, kkt_max_itr=100, cd_tol=1e-4, verbose=1, **kwargs):
    
    n,p,X,y,S_diag = preprocess(X, y)

    if maxSuppSize is None:
        maxSuppSize = p
    _sol_str = 'beta sol_time support cost'
    Solution = namedtuple('Solution', _sol_str)
    sols = [] 
    terminate = False
    iteration_num = 0
    M_grid = []
    if M is not None:
        M_fixed = True
    else:
        M_fixed = False
    if verbose:
        print("BnB Started.")
    kwargs['verbose'] = (verbose>1)
    heuristic_solver =  kwargs.get("upper_solver", "L0L2_ASCD")
       
    if lambda0_grid is not None:
        lambda0_grid = sorted(lambda0_grid, reverse = True)
        beta = None
        while not terminate:
            l0 = lambda0_grid[iteration_num]
            st  = time()
            if verbose:
                print(l0,lambda2)
            beta, cost, Xb, support = heuristic_solve(loss_name, X, y, l0, lambda2, M=np.inf if not M_fixed else M, \
                                                      delta=delta, solver=heuristic_solver, upper_support_heuristic="all", beta=beta, z=None, \
                                                      S_diag=S_diag, rel_tol=rel_tol, kkt_max_itr=kkt_max_itr, cd_max_itr=cd_max_itr, cd_tol=cd_tol, \
                                                      timing=False, maxtime=np.inf, verbose=(verbose>2))
            if not M_fixed:
                M = np.max(np.abs(beta))*M_multiplier
                if verbose:
                    print("M = ", M)
            M_grid.append(M)
            tree = BNBTree(loss_name,X,y,delta)
            #sol = tree.solve(l0,lambda2,M,beta,**kwargs)
            sol = tree.solve(l0,lambda2,M,beta,rel_tol=rel_tol, kkt_max_itr=kkt_max_itr, cd_max_itr=cd_max_itr, cd_tol=cd_tol,**kwargs)
            sols.append(sol)
            beta = sol.beta
            support = set(np.where(beta)[0])
            if verbose:
                print("Iteration: " + str(iteration_num) + ". Number of non-zeros: ",len(support))
                
            iteration_num += 1
            if iteration_num == len(lambda0_grid):
                terminate = True
            if len(support) >= maxSuppSize:
                terminate = True
                lambda0_grid = lambda0_grid[:iteration_num]
    else:
        beta, Xb = trivial_soln(X)
        support = set()
        lambda0_grid = []
        eps_factor = min(1e-3, (1-scaleDownFactor)*1e-2)
        while not terminate:
            if iteration_num == 0:
                l0 = compute_next_lambda(loss_name,X,y,lambda2,np.inf,S_diag,beta,Xb,support,delta)*(1-eps_factor)
            else:
                l0 = min(l0*scaleDownFactor, compute_next_lambda(loss_name,X,y,lambda2,np.inf,S_diag,beta,Xb,support,delta)*(1-eps_factor))
            lambda0_grid.append(l0)
            st  = time()
            if verbose:
                print(l0,lambda2)
            beta, cost, Xb, support = heuristic_solve(loss_name, X, y, l0, lambda2, M=np.inf if not M_fixed else M, \
                                                      delta=delta, solver=heuristic_solver, upper_support_heuristic="all", beta=beta, z=None, \
                                                      S_diag=S_diag, rel_tol=rel_tol, kkt_max_itr=kkt_max_itr, cd_max_itr=cd_max_itr, cd_tol=cd_tol, \
                                                      timing=False, maxtime=np.inf, verbose=(verbose>2))
            if not M_fixed:
                M = np.max(np.abs(beta))*M_multiplier
                if verbose:
                    print("M = ", M)
            M_grid.append(M)
            tree = BNBTree(loss_name,X,y,delta)
            #sol = tree.solve(l0,lambda2,M,beta,**kwargs)
            sol = tree.solve(l0,lambda2,M,beta,rel_tol=rel_tol, kkt_max_itr=kkt_max_itr, cd_max_itr=cd_max_itr, cd_tol=cd_tol,**kwargs)
            sols.append(sol)
            beta = sol.beta
            support = set(np.where(beta)[0])
            if verbose:
                print("Iteration: " + str(iteration_num) + ". Number of non-zeros: ",len(support))
            iteration_num += 1
            if iteration_num == n_lambda0:
                terminate = True
            if len(support) >= maxSuppSize:
                terminate = True
            if l0 == 0:
                terminate = True
            
    return lambda0_grid, M_grid, sols