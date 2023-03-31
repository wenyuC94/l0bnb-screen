import copy
from time import time
from collections import namedtuple

import numpy as np
from numba.typed import List
from numba import njit

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

from ._coordinate_descent import cd_loop, cd
from ._proximal_gradient import pg
from ._cost import get_primal_cost, get_dual_cost_given_loss_value
from ..utils import get_ratio_threshold, compute_relative_gap, trivial_soln, set2arr, nb_set2arr, is_integral, skl_svd #, active_set_to_support
from ...loss_utils import compute_grad
from ..screening import compute_lower_bounds_for_fixed_z_with_optimal_solution, \
                       compute_lower_bounds_for_fixed_z_with_approximate_solution, \
                       compute_lower_bounds_for_fixed_z_with_approximate_solution_new

EPSILON = np.finfo('float').eps



# this one need to be refactored
def _initial_active_set(X, y, beta, zlb, zub, upper_active_set_mask, sp_ratio = 0.2):
    p = X.shape[1]
    corr = np.corrcoef(X,y[:,None],rowvar=False)[:-1,-1]
    sp_thres = int(sp_ratio*p)
    if sp_thres < p:
        argpart = np.argpartition(-np.abs(corr), sp_thres)[:sp_thres]
    else:
        argpart = np.arange(p)
        
    active_set = set(argpart)
    
    active_set = active_set | set(np.where(zlb==1)[0])
    
    active_set = active_set | set(np.where(np.abs(beta)>EPSILON*1e10)[0])
    
    active_set = active_set - set(np.where(~upper_active_set_mask)[0])
    
    active_set = np.array(sorted(active_set),dtype=int)
    return active_set



@njit(cache=True)
def _refined_initial_active_set(loss_name, X, y, beta, l0, l2, M, ratio, threshold, S_diag, zlb, zub, active_set, support, Xb, delta):
    support.clear()
    num_of_similar_supports = 0
    delta_supp = 0
    while num_of_similar_supports < 3:
        delta_supp = 0
        beta, Xb = cd_loop(loss_name, X, y, beta, l0, l2, M, ratio, threshold, S_diag, zlb, zub, active_set, Xb, delta)
        for i in active_set:
            if (beta[i]!=0) and (i not in support):
                support.add(i)
                delta_supp += 1
        if delta_supp == 0:
            num_of_similar_supports += 1
        else:
            num_of_similar_supports = 0
    return support, beta, Xb


def _initialize_active_set_algo(loss_name, X, y, l0, l2, M, ratio, threshold, S_diag, ones_index, zeros_index, upper_active_set_mask, warm_start, delta=1., sp_ratio=0.2):
    p = X.shape[1]
    ones_index = set2arr(ones_index)
    zeros_index = set2arr(zeros_index)
    zlb = np.zeros(p)
    zlb[ones_index] = 1
    zub = np.ones(p)
    zub[zeros_index] = 0
    if S_diag is None:
        S_diag = np.linalg.norm(X, axis=0)**2
    if warm_start is not None:
        support, beta = warm_start['support'], np.copy(warm_start['beta'])
        beta[zeros_index] = 0
        beta[~upper_active_set_mask] = 0
        support = (support - set(zeros_index) - set(np.where(~upper_active_set_mask)[0])) | set(ones_index)
        Xb = warm_start.get('Xb', X@beta)
        active_set = set2arr(support)
    else:
        beta, Xb = trivial_soln(X)
        active_set = _initial_active_set(X, y, beta, zlb, zub, upper_active_set_mask, sp_ratio)
#         support = {0}
#         support, beta, Xb = _refined_initial_active_set(loss_name, X, y, beta, l0, l2, M, ratio, threshold, S_diag, zlb, zub, active_set, support, Xb, delta)
        support = set(active_set)
        
    return beta, Xb, support, zub, zlb, S_diag

def _initialize_active_set_algo_ASPG(loss_name, X, y, l0, l2, M, ratio, threshold, L, ones_index, zeros_index, upper_active_set_mask, warm_start, delta=1.):
    p = X.shape[1]
    ones_index = set2arr(ones_index)
    zeros_index = set2arr(zeros_index)
    zlb = np.zeros(p)
    zlb[ones_index] = 1
    zub = np.ones(p)
    zub[zeros_index] = 0
    if L is None:
        if loss_name == 'lstsq':
            L = skl_svd(X)**2
        elif loss_name == 'logistic':
            L = 0.25*skl_svd(X)**2
        elif loss_name == 'sqhinge':
            L = 2*skl_svd(X)**2
        elif loss_name == 'hbhinge':
            L = skl_svd(X)**2/delta
    if warm_start is not None:
        support, beta = warm_start['support'], np.copy(warm_start['beta'])
        beta[zeros_index] = 0
        beta[~upper_active_set_mask] = 0
        support = (support - set(zeros_index) - set(np.where(~upper_active_set_mask)[0])) | set(ones_index)
        Xb = warm_start.get('Xb', X@beta)
        active_set = set2arr(support)
    else:
        beta, Xb = trivial_soln(X)
        active_set = _initial_active_set(X, y, beta, zlb, zub, upper_active_set_mask)
#         support = {0}
#         support, beta, Xb = _refined_initial_active_set(loss_name, X, y, beta, l0, l2, M, ratio, threshold, S_diag, zlb, zub, active_set, support, Xb, delta)
        support = set(active_set)
        
    return beta, Xb, support, zub, zlb, L



@njit(cache=True, parallel=True)
def _above_threshold_indices(loss_name, X, y, beta, Xb, threshold, zub, upper_active_set_mask, delta):
    grad = compute_grad(loss_name, y, X, beta, Xb, delta)
    above_threshold = np.where((zub*np.abs(grad)-threshold > 0)&upper_active_set_mask)[0]
    return above_threshold, grad

@njit
def relax_ASCD(loss_name, X, y, l0, l2, M, zlb, zub, ratio, threshold, S_diag, beta, Xb, active_set, upper_active_set_mask, delta=1., \
                rel_tol=1e-4, cd_max_itr=100, kkt_max_itr=100, verbose=False):
    '''
    Solve the relaxation problem with active set coordinate descent.
    Input requirement:
        l0,l2,M >= 0
        zlb, zub are vectors with only zeros and ones
        ratio, threshold should be get_ratio_threshold(l0, l2, M)   
        S_diag should be np.linalg.norm(X,axis=0)**2 
        Xb should be X@beta
        beta[~active_set] should be 0 (i.e., beta[i]>0 => i in active_set)
        zlb[~active_set] should be 0 (i.e., zlb[i]=1 => i in active_set)
        upper_active_set_mask[active_set] should be True (i.e., upper_active_set_mask[i]=True => i not in active_set)
    '''
    support = set(active_set)
    cost, loss, _ = get_primal_cost(loss_name, X, y, beta, Xb, l0, l2, M, ratio, threshold, zlb, zub, active_set, delta)
    cd_tol = rel_tol / 2
    if verbose:
        print('cost', cost)
    curiter = 0
    while curiter < kkt_max_itr:
        beta, cost, Xb, loss = cd(loss_name, X, y, beta, cost, l0, l2, M, ratio, threshold, S_diag, zlb, zub, active_set, Xb, cd_tol, cd_max_itr, verbose, delta)
        if verbose:
            print("iter", curiter+1)
            print("cost", cost)
        above_threshold, grad = _above_threshold_indices(loss_name, X, y, beta, Xb, threshold, zub, upper_active_set_mask, delta)
        outliers = list(set(above_threshold) - support)
        if len(outliers) == 0:
            if verbose:
                print("no outliers, computing relative accuracy...")
            dual_cost = get_dual_cost_given_loss_value(X, y, beta, loss, grad, l0, l2, M, ratio, threshold, zlb, zub, active_set)
            if verbose:
                print("dual", dual_cost)
            if compute_relative_gap(cost, dual_cost) < rel_tol:
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
        
        #print("outlier support is ",set(outliers))
    if curiter == kkt_max_itr:
        print('Maximum KKT check iterations reached, increase kkt_max_itr '
              'to avoid this warning')
    
    
    active_set = np.where(beta)[0]
    cost, loss, z = get_primal_cost(loss_name, X, y, beta, Xb, l0, l2, M, ratio, threshold, zlb, zub, active_set, delta)
    
    return beta, z, cost, dual_cost, Xb, loss, grad, active_set

@njit
def _relax_ASCD_for_node_base(loss_name, X, y, l0, l2, M, zlb, zub, ratio, threshold, S_diag, beta, Xb, active_set, upper_active_set_mask, delta=1.,
               rel_tol=1e-4, tree_upper_bound=np.inf, mio_gap=0, check_if_integral=True, cd_max_itr=100, kkt_max_itr=100, verbose=False):
    '''
    Solve the relaxation problem with active set coordinate descent for the BnB node purpose.
    Input requirement:
        l0,l2,M >= 0
        zlb, zub are vectors with only zeros and ones
        ratio, threshold should be get_ratio_threshold(l0, l2, M)   
        S_diag should be np.linalg.norm(X,axis=0)**2 
        Xb should be X@beta
        beta[~active_set] should be 0 (i.e., beta[i]>0 => i in active_set)
        zlb[~active_set] should be 0 (i.e., zlb[i]=1 => i in active_set)
        upper_active_set_mask[active_set] should be True (i.e., upper_active_set_mask[i]=True => i not in active_set)
    '''
    support = set(active_set)
    cost, loss, _ = get_primal_cost(loss_name, X, y, beta, Xb, l0, l2, M, ratio, threshold, zlb, zub, active_set, delta)
    cd_tol = rel_tol / 2
    if verbose:
        print('cost', cost)
    curiter = 0
    while curiter < kkt_max_itr:
        beta, cost, Xb, loss = cd(loss_name, X, y, beta, cost, l0, l2, M, ratio, threshold, S_diag, zlb, zub, active_set, Xb, cd_tol, cd_max_itr, verbose, delta)
        if verbose:
            print("iter", curiter+1)
            print("cost", cost)
        above_threshold, grad = _above_threshold_indices(loss_name, X, y, beta, Xb, threshold, zub, upper_active_set_mask, delta)
        outliers = list(set(above_threshold) - support)
        if len(outliers) == 0:
            if verbose:
                print("no outliers, computing relative accuracy...")
            dual_cost = get_dual_cost_given_loss_value(X, y, beta, loss, grad, l0, l2, M, ratio, threshold, zlb, zub, active_set)
            if verbose:
                print("dual", dual_cost)
                
            if not check_if_integral or tree_upper_bound == np.inf:
                cur_gap = -2
                # tree_upper_bound = dual_cost + 1
            else:
                cur_gap = compute_relative_gap(tree_upper_bound,cost)
            
            if cur_gap < mio_gap and tree_upper_bound > dual_cost:
                if (compute_relative_gap(cost, dual_cost) < rel_tol) or \
                        (cd_tol < 1e-8 and check_if_integral):
                    break
                else:
                    cd_tol /= 100
            else:
                break
        
        support = support | set(outliers)
        if len(support) == 0:
            active_set = np.full(0,0)
        else:
            active_set = nb_set2arr(support)
        curiter += 1
    if curiter == kkt_max_itr:
        print('Maximum KKT check iterations reached, increase kkt_max_itr '
              'to avoid this warning')
        
    active_set = np.where(beta)[0]
    cost, loss, z = get_primal_cost(loss_name, X, y, beta, Xb, l0, l2, M, ratio, threshold, zlb, zub, active_set, delta)
    
    return beta, z, cost, dual_cost, Xb, loss, grad, active_set


def relax_ASCD_for_node(loss_name, X, y, l0, l2, M, zlb, zub, ratio, threshold, S_diag, beta, Xb, active_set, upper_active_set_mask, delta=1.,
               rel_tol=1e-4, int_tol=1e-4, tree_upper_bound=np.inf, mio_gap=0, check_if_integral=True, cd_max_itr=100, kkt_max_itr=100, verbose=False):
    '''
    Solve the relaxation problem with active set coordinate descent for the BnB node purpose.
    Input requirement:
        l0,l2,M >= 0
        zlb, zub are vectors with only zeros and ones
        ratio, threshold should be get_ratio_threshold(l0, l2, M)   
        S_diag should be np.linalg.norm(X,axis=0)**2 
        Xb should be X@beta
        beta[~active_set] should be 0 (i.e., beta[i]>0 => i in active_set)
        zlb[~active_set] should be 0 (i.e., zlb[i]=1 => i in active_set)
        upper_active_set_mask[active_set] should be True (i.e., upper_active_set_mask[i]=True => i not in active_set)
    '''
    beta, z, cost, dual_cost, Xb, loss, grad, active_set = \
        _relax_ASCD_for_node_base(loss_name, X, y, l0, l2, M, zlb, zub, ratio, threshold, S_diag, beta, Xb, active_set, upper_active_set_mask, delta,
               rel_tol, tree_upper_bound, mio_gap, check_if_integral, cd_max_itr, kkt_max_itr, verbose)
    prim_dual_gap = compute_relative_gap(cost, dual_cost)
    
    if check_if_integral and prim_dual_gap > rel_tol and is_integral(z, int_tol):
        if verbose:
            print("integral solution obtained: perform exact optimization")
        beta, z, cost, dual_cost, Xb, loss, grad, active_set = _relax_ASCD_for_node_base(loss_name, X, y, l0, l2, M, zlb, zub, ratio, threshold, S_diag, beta, Xb, active_set, upper_active_set_mask, delta,\
                                                                   rel_tol, tree_upper_bound, mio_gap, False, cd_max_itr, kkt_max_itr, verbose)
    
    return beta, z, cost, dual_cost, Xb, loss, grad, active_set

def relax_ASCD_solve(loss_name, X, y, l0, l2, M, ones_index, zeros_index, upper_active_set_mask = None, delta=1.,
                ratio=None, threshold=None, S_diag=None, warm_start=None, 
                rel_tol=1e-4, int_tol=1e-4, tree_upper_bound=np.inf, mio_gap=0, check_if_integral=True, ascd_heuristics=True, 
                cd_max_itr=100, kkt_max_itr=100, compute_fixed_lbs=False, fixed_lb_method='exact', verbose=False, sp_ratio = 0.2):
    st = time()
    
    _sol_str = 'primal_value dual_value support beta z Xb loss sol_time lower_bounds'
    Solution = namedtuple('Solution', _sol_str)
    
    if upper_active_set_mask is None:
        upper_active_set_mask = np.full(X.shape[1], True)
    
    if ratio is None or threshold is None:
        ratio, threshold = get_ratio_threshold(l0, l2, M)
    
    beta, Xb, support, zub, zlb, S_diag = \
        _initialize_active_set_algo(loss_name, X, y, l0, l2, M, \
                                    ratio, threshold, S_diag, ones_index, zeros_index, upper_active_set_mask, warm_start, delta, sp_ratio)
    
    active_set = set2arr(support)
    
    if ascd_heuristics:
        beta, z, cost, dual_cost, Xb, loss, grad, active_set = \
            relax_ASCD_for_node(loss_name, X, y, l0, l2, M, zlb, zub, ratio, threshold, S_diag, beta, Xb, active_set, upper_active_set_mask, delta,
                   rel_tol, int_tol, tree_upper_bound, mio_gap, check_if_integral, cd_max_itr, kkt_max_itr, verbose)
    else:
        beta, z, cost, dual_cost, Xb, loss, grad, active_set = \
            relax_ASCD(loss_name, X, y, l0, l2, M, zlb, zub, ratio, threshold, S_diag, beta, Xb, active_set, upper_active_set_mask, delta, rel_tol, cd_max_itr, kkt_max_itr, verbose)
        
    sol_time = time() - st
    
    if compute_fixed_lbs:
        assert fixed_lb_method in {'exact','approx','approx-new'}
        if fixed_lb_method == 'exact':
            bounds = compute_lower_bounds_for_fixed_z_with_optimal_solution(loss_name, X, y, beta, z, zlb, zub, l0, l2, M, Xb, dual_cost, grad, delta)
        elif fixed_lb_method == 'approx':
            bounds = compute_lower_bounds_for_fixed_z_with_approximate_solution(loss_name, X, y, beta, z, zlb, zub, l0, l2, M, Xb, dual_cost, grad, loss, delta)
        elif fixed_lb_method == 'approx-new':
            bounds = compute_lower_bounds_for_fixed_z_with_approximate_solution_new(loss_name, X, y, beta, z, zlb, zub, l0, l2, M, Xb,  dual_cost, grad, ratio, threshold, delta)
    else:
        bounds = None
    
    sol = Solution(primal_value=cost, dual_value=dual_cost, support=set(active_set), beta=beta, z=z, Xb = Xb, loss=loss, sol_time=sol_time, lower_bounds=bounds)
    
    return sol




@njit
def relax_ASPG(loss_name, X, y, l0, l2, M, zlb, zub, ratio, threshold, L, beta, Xb, active_set, upper_active_set_mask, delta=1., \
                rel_tol=1e-4, cd_max_itr=100, kkt_max_itr=100, verbose=False):
    '''
    Solve the relaxation problem with active set proximal gradient.
    Input requirement:
        l0,l2,M >= 0
        zlb, zub are vectors with only zeros and ones
        ratio, threshold should be get_ratio_threshold(l0, l2, M)   
        Xb should be X@beta
        beta[~active_set] should be 0 (i.e., beta[i]>0 => i in active_set)
        zlb[~active_set] should be 0 (i.e., zlb[i]=1 => i in active_set)
        upper_active_set_mask[active_set] should be True (i.e., upper_active_set_mask[i]=True => i not in active_set)
    '''
    support = set(active_set)
    cost, loss, _ = get_primal_cost(loss_name, X, y, beta, Xb, l0, l2, M, ratio, threshold, zlb, zub, active_set, delta)
    cd_tol = rel_tol / 2
    if verbose:
        print('cost', cost)
    curiter = 0
    while curiter < kkt_max_itr:
        beta, cost, Xb, loss = pg(loss_name, X, y, beta, cost, l0, l2, M, ratio, threshold, L, zlb, zub, active_set, Xb, cd_tol, cd_max_itr, verbose, delta)
        if verbose:
            print("iter", curiter+1)
            print("cost", cost)
        above_threshold, grad = _above_threshold_indices(loss_name, X, y, beta, Xb, threshold, zub, upper_active_set_mask, delta)
        outliers = list(set(above_threshold) - support)
        if len(outliers) == 0:
            if verbose:
                print("no outliers, computing relative accuracy...")
            dual_cost = get_dual_cost_given_loss_value(X, y, beta, loss, grad, l0, l2, M, ratio, threshold, zlb, zub, active_set)
            if verbose:
                print("dual", dual_cost)
            if compute_relative_gap(cost, dual_cost) < rel_tol:
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
    
    
    active_set = np.where(beta)[0]
    cost, loss, z = get_primal_cost(loss_name, X, y, beta, Xb, l0, l2, M, ratio, threshold, zlb, zub, active_set, delta)
    
    return beta, z, cost, dual_cost, Xb, loss, grad, active_set


def relax_ASPG_solve(loss_name, X, y, l0, l2, M, ones_index, zeros_index, upper_active_set_mask = None, delta=1.,
                ratio=None, threshold=None, L=None, warm_start=None, 
                rel_tol=1e-4, int_tol=1e-4, tree_upper_bound=np.inf, mio_gap=0, check_if_integral=True, ascd_heuristics=True, 
                cd_max_itr=100, kkt_max_itr=100, compute_fixed_lbs=False, fixed_lb_method='exact', verbose =False):
    st = time()
    
    _sol_str = 'primal_value dual_value support beta z Xb loss sol_time lower_bounds'
    Solution = namedtuple('Solution', _sol_str)
    
    if upper_active_set_mask is None:
        upper_active_set_mask = np.full(X.shape[1], True)
    
    if ratio is None or threshold is None:
        ratio, threshold = get_ratio_threshold(l0, l2, M)
    
    beta, Xb, support, zub, zlb, L = \
        _initialize_active_set_algo_ASPG(loss_name, X, y, l0, l2, M, \
                                    ratio, threshold, L, ones_index, zeros_index,\
                                    upper_active_set_mask, warm_start, delta)
    
    active_set = set2arr(support)
    
    beta, z, cost, dual_cost, Xb, loss, grad, active_set = \
            relax_ASPG(loss_name, X, y, l0, l2, M, zlb, zub, ratio, threshold, L, beta, Xb, active_set, upper_active_set_mask, delta, rel_tol, cd_max_itr, kkt_max_itr, verbose)
        
    sol_time = time() - st
    
    if compute_fixed_lbs:
        assert fixed_lb_method in {'exact','approx','approx-new'}
        if fixed_lb_method == 'exact':
            bounds = compute_lower_bounds_for_fixed_z_with_optimal_solution(loss_name, X, y, beta, z, zlb, zub, l0, l2, M, Xb, dual_cost, grad, delta)
        elif fixed_lb_method == 'approx':
            bounds = compute_lower_bounds_for_fixed_z_with_approximate_solution(loss_name, X, y, beta, z, zlb, zub, l0, l2, M, Xb, dual_cost, grad, loss, delta)
        elif fixed_lb_method == 'approx-new':
            bounds = compute_lower_bounds_for_fixed_z_with_approximate_solution_new(loss_name, X, y, beta, z, zlb, zub, l0, l2, M, Xb,  dual_cost, grad, ratio, threshold, delta)
    else:
        bounds = None
    
    sol = Solution(primal_value=cost, dual_value=dual_cost, support=set(active_set), beta=beta, z=z, Xb = Xb, loss=loss, sol_time=sol_time, lower_bounds=bounds)
    
    return sol
