import copy
from time import time
from collections import namedtuple
import math

import numpy as np
from numba.typed import List
from numba import njit, objmode

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

from ._coordinate_descent import L0L2_CD, L0L2_CD_loop
from ._cost import get_L0L2_cost
from ..utils import compute_relative_gap, set2arr, nb_set2arr, trivial_soln
from ...loss_utils import compute_grad_Lipschitz_sigma, compute_grad

EPSILON = np.finfo('float').eps

def _initial_active_set(X, y, beta, upper_active_set_mask):
    p = X.shape[1]
    corr = np.corrcoef(X,y[:,None],rowvar=False)[:-1,-1]
    argpart = np.argpartition(-np.abs(corr), int(0.2*p))[:int(0.2*p)]
    active_set = set(argpart)
    
    active_set = active_set | set(np.where(np.abs(beta)>EPSILON*1e10)[0])
    
    active_set = active_set - set(np.where(~upper_active_set_mask)[0])
    
    active_set = np.array(sorted(active_set),dtype=int)
    return active_set


@njit
def _refined_initial_active_set(loss_name, X, y, beta, l0, l2, M, S_diag, active_set, support, Xb, delta=1.):
    support.clear()
    num_of_similar_supports = 0
    delta_supp = 0
    while num_of_similar_supports < 3:
        delta_supp = 0
        beta, Xb = L0L2_CD_loop(loss_name, X, y, beta, l0, l2, M, S_diag, active_set, Xb, delta)
        for i in active_set:
            if (beta[i]!=0) and (i not in support):
                support.add(i)
                delta_supp += 1
        if delta_supp == 0:
            num_of_similar_supports += 1
        else:
            num_of_similar_supports = 0
    return support, beta, Xb


def _initialize_active_set_algo(loss_name, X, y, l0, l2, M, S_diag, upper_active_set_mask, warm_start, delta=1.):
    p = X.shape[1]
    if S_diag is None:
        S_diag = np.linalg.norm(X, axis=0)**2
    if warm_start is not None:
        support, beta = warm_start['support'], np.copy(warm_start['beta'])
        beta[~upper_active_set_mask] = 0
        support = support - set(np.where(~upper_active_set_mask)[0])
        Xb = warm_start.get('Xb', X@beta)
        active_set = set2arr(support)
    else:
        beta, Xb = trivial_soln(X)
        active_set = _initial_active_set(X, y, beta, upper_active_set_mask)
        support = {0}
        support, beta, Xb = _refined_initial_active_set(loss_name, X, y, beta, l0, l2, M, S_diag, active_set, support, Xb, delta)
        
    return beta, Xb, support, S_diag


@njit(parallel=True)
def _above_threshold_indices(loss_name,l0,l2,M,X,y,S_diag,beta,Xb,upper_active_set_mask,delta=1.):
    sigma = compute_grad_Lipschitz_sigma(loss_name)
    grad = compute_grad(loss_name, y, X, beta, Xb, delta)
    a = sigma*S_diag/2
    b = -a*2*beta+grad
    criterion = np.where(np.abs(b)/2/(a+l2)<=M, 4*l0*(a+l2)-b**2, a*M**2-np.abs(b)*M+l0+l2*M**2)
    above_threshold = np.where((criterion<0)&(upper_active_set_mask))[0]
    return above_threshold


# @njit
# def L0L2_ASCD(loss_name, X, y, l0, l2, M, S_diag, beta, Xb, active_set, upper_active_set_mask, delta=1., \
#                 cd_tol=1e-4, cd_max_itr=100, rel_tol=1e-6, kkt_max_itr=100, verbose=False):
#     support = set(active_set)
#     cost = get_L0L2_cost(loss_name, X, y, beta, Xb, l0, l2, M, active_set, delta)
#     old_cost = cost
#     if verbose:
#         print("cost", cost)
#     curiter = 0
#     while curiter < kkt_max_itr:
#         beta, cost, Xb = L0L2_CD(loss_name, X, y, beta, cost, l0, l2, M, S_diag, active_set, Xb, cd_tol, cd_max_itr, verbose, delta)
#         if verbose:
#             print("iter", curiter+1)
#             print("cost", cost)
#         above_threshold = _above_threshold_indices(loss_name,l0,l2,M,X,y,S_diag,beta,Xb,upper_active_set_mask,delta)
#         outliers = list(set(above_threshold) - support)
#         if len(outliers) == 0:
#             if verbose:
#                 print("no outliers, computing relative accuracy...")
#             if compute_relative_gap(cost, old_cost) < rel_tol or cd_tol < 1e-8:
#                 break
#             else:
#                 cd_tol /= 100
#                 old_cost = cost
        
#         support = support | set(outliers)
#         if len(support) == 0:
#             active_set = np.full(0,0)
#         else:
#             active_set = nb_set2arr(support)
#         curiter += 1
#     if curiter == kkt_max_itr:
#         print('Maximum KKT check iterations reached, increase kkt_max_itr '
#               'to avoid this warning')
#     active_set = np.where(beta)[0]
#     return beta, cost, Xb, active_set

@njit
def L0L2_ASCD(loss_name, X, y, l0, l2, M, S_diag, beta, Xb, active_set, upper_active_set_mask, delta=1., \
                cd_tol=1e-4, cd_max_itr=100, rel_tol=1e-6, kkt_max_itr=100, timing=False, maxtime=np.inf, verbose=False):
    start_time = 0.
    end_time = 0.
    if timing:
        with objmode(start_time='f8'):
            start_time = time()
    support = set(active_set)
    cost = get_L0L2_cost(loss_name, X, y, beta, Xb, l0, l2, M, active_set, delta)
    old_cost = cost
    if verbose:
        print("cost", cost)
    curiter = 0
    while curiter < kkt_max_itr:
        if timing:
            with objmode(end_time='f8'):
                end_time = time()
            if end_time - start_time > maxtime:
                break
        beta, cost, Xb = L0L2_CD(loss_name, X, y, beta, cost, l0, l2, M, S_diag, active_set, Xb, cd_tol, cd_max_itr, verbose, delta)
        if verbose:
            print("iter", curiter+1)
            print("cost", cost)
        above_threshold = _above_threshold_indices(loss_name,l0,l2,M,X,y,S_diag,beta,Xb,upper_active_set_mask,delta)
        outliers = list(set(above_threshold) - support)
        if len(outliers) == 0:
            if verbose:
                print("no outliers, computing relative accuracy...")
            if compute_relative_gap(cost, old_cost) < rel_tol or cd_tol < 1e-8:
                break
            else:
                cd_tol /= 100
                old_cost = cost
        
        support = support | set(outliers)
        if len(support) == 0:
            active_set = np.full(0,0)
        else:
            active_set = nb_set2arr(support)
        curiter += 1
    if curiter == kkt_max_itr:
        print('Maximum KKT check iterations reached, increase kkt_max_itr '
              'to avoid this warning')
    if timing and end_time - start_time > maxtime:
        print('Time limit reached, increase maxtime '
              'to avoid this warning')
    active_set = np.where(beta)[0]
    return beta, cost, Xb, active_set


def L0L2_ASCD_solve(loss_name, X, y, l0, l2, M, upper_active_set_mask=None, delta=1., S_diag=None, warm_start=None, \
                    cd_tol=1e-4, cd_max_itr=100, rel_tol=1e-6, kkt_max_itr=100, timing=False, maxtime=np.inf, verbose=False):
    
    n,p = X.shape
    if upper_active_set_mask is None:
        upper_active_set_mask = np.full(p,True)
    
    beta, Xb, support, S_diag = _initialize_active_set_algo(loss_name, X, y, l0, l2, M, S_diag, upper_active_set_mask, warm_start, delta)
    active_set = set2arr(support)
    
    beta, cost, Xb, active_set =  L0L2_ASCD(loss_name, X, y, l0, l2, M, S_diag, beta, Xb, active_set, upper_active_set_mask, delta, \
                                            cd_tol, cd_max_itr, rel_tol, kkt_max_itr, timing, maxtime, verbose)
    
    return beta, cost, Xb, set(active_set)


##########
# CDPSI
##########

##########
# ASCD_PSI
##########

##########
# AS_CDPSI
##########