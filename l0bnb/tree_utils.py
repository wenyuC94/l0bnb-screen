import copy
import sys
import numpy as np

from .node import Node, NodeStatus
from .solvers.relaxation._coordinate_descent import cd
from .solvers.relaxation._cost import get_dual_cost_given_loss_name, get_primal_cost
from .solvers.utils import set2arr

def max_fraction_branching(z, ones_index, zeros_index, tol):
    p = len(z)
    relax_index = np.array(list(set(range(p))-ones_index-zeros_index))
    if len(relax_index)==0:
        return -1
    z_relax = z[relax_index]
    idx = np.argmin(np.abs(z_relax-0.5))
    return relax_index[idx]

def max_z_branching(beta, ones_index, zeros_index, tol):
    p = len(beta)
    relax_index = np.array(list(set(range(p))-ones_index-zeros_index))
    if len(relax_index)==0:
        return -1
    beta_relax = beta[relax_index]
    idx = np.argmax(np.abs(beta_relax))
    return relax_index[idx]

def max_lb_branching(bounds, ones_index, zeros_index, tol):
    p = bounds.shape[0]
    relax_index = np.array(list(set(range(p))-ones_index-zeros_index))
    if len(relax_index)==0:
        return -1
    maxlbs = np.max(bounds,axis=1)
    #print(maxlbs)
    maxlbs_relax = maxlbs[relax_index]
    idx = np.argmax(maxlbs_relax)
    #print(idx)
    return relax_index[idx]

def min_lb_branching(bounds, ones_index, zeros_index, tol):
    p = bounds.shape[0]
    relax_index = np.array(list(set(range(p))-ones_index-zeros_index))
    if len(relax_index)==0:
        return -1
    minlbs = np.min(bounds,axis=1)
    minlbs_relax = minlbs[relax_index]
    idx = np.argmin(minlbs_relax)
    return relax_index[idx]

def strong_branching(loss_name, X, y, beta, z, cost, dual_cost, l0, l2, M, ratio, threshold, S_diag, ones_index, zeros_index, tol, act_set, mu=1., maxiter=3000):
    p = len(z)
    if act_set is None:
        active_set = np.array(range(p))
    else:
        active_set = np.array(list(act_set))
    relax_index = np.array(list(set(np.where((z>0)&(z<1))[0])-ones_index-zeros_index))
    if len(relax_index)==0:
        return -1
    ones_index = set2arr(ones_index)
    zeros_index = set2arr(zeros_index)
    zlb = np.zeros(p)
    zlb[ones_index] = 1
    zub = np.ones(p)
    zub[zeros_index] = 0
    new_bounds = np.full((len(relax_index),2),cost)
    for i in range(len(relax_index)):
        zub[relax_index[i]] = 0
        beta0,cost0,Xb0,_ = cd(loss_name, X, y, beta, cost, l0, l2, M, ratio, threshold, S_diag, zlb, zub,\
           active_set=active_set, Xb=X@beta, rel_tol=1e-8, maxiter=maxiter, verbose=False, delta=1.)
        # new_bound0 = get_dual_cost_given_loss_name(loss_name, X, y, beta0, Xb0, l0, l2, M, ratio,\
        #                                 threshold, zlb, zub, active_set=np.array(range(p)), delta=1.)
        # new_bounds[i,0] = max(dual_cost, new_bound0)
        new_bounds[i,0] = cost0
        zub[relax_index[i]] = 1
        zlb[relax_index[i]] = 1
        beta1,cost1,Xb1,_ = cd(loss_name, X, y, beta, cost, l0, l2, M, ratio, threshold, S_diag, zlb, zub,\
           active_set=active_set, Xb=X@beta, rel_tol=1e-8, maxiter=maxiter, verbose=False, delta=1.)
        # new_bound1 = get_dual_cost_given_loss_name(loss_name, X, y, beta1, Xb1, l0, l2, M, ratio,\
        #                                 threshold, zlb, zub, active_set=np.array(range(p)), delta=1.)
        # new_bounds[i,1] = max(dual_cost, new_bound1)
        new_bounds[i,1] = cost1
        zlb[relax_index[i]] = 0      # change zlb, zub back to original
    #avg_incr = mu*np.min(new_bounds-dual_cost,axis=1)+(1-mu)*np.max(new_bounds-dual_cost,axis=1)
    avg_incr = mu*np.min(new_bounds-cost,axis=1)+(1-mu)*np.max(new_bounds-cost,axis=1)
    idx = np.argmax(avg_incr)
    return relax_index[idx]

def full_strong_branching(loss_name, X, y, beta, z, cost, dual_cost, l0, l2, M, ratio, threshold, S_diag, ones_index, zeros_index, tol, act_set, epsilon=1e-6, maxiter=3000):
    p = len(z)
    if act_set is None:
        active_set = np.array(range(p))
    else:
        active_set = np.array(list(act_set))
    relax_index = np.array(list(set(np.where((z>0)&(z<1))[0])-ones_index-zeros_index))
    if len(relax_index)==0:
        return -1
    ones_index = set2arr(ones_index)
    zeros_index = set2arr(zeros_index)
    zlb = np.zeros(p)
    zlb[ones_index] = 1
    zub = np.ones(p)
    zub[zeros_index] = 0
    new_bounds = np.full((len(relax_index),2),cost)
    for i in range(len(relax_index)):
        zub[relax_index[i]] = 0
        beta0,cost0,Xb0,_ = cd(loss_name, X, y, beta, cost, l0, l2, M, ratio, threshold, S_diag, zlb, zub,\
           active_set=active_set, Xb=X@beta, rel_tol=1e-8, maxiter=maxiter, verbose=False, delta=1.)
        # new_bound0 = get_dual_cost_given_loss_name(loss_name, X, y, beta0, Xb0, l0, l2, M, ratio,\
        #                                 threshold, zlb, zub, active_set=np.array(range(p)), delta=1.)
        # new_bounds[i,0] = max(dual_cost, new_bound0)
        new_bounds[i,0] = cost0
        zub[relax_index[i]] = 1
        zlb[relax_index[i]] = 1
        beta1,cost1,Xb1,_ = cd(loss_name, X, y, beta, cost, l0, l2, M, ratio, threshold, S_diag, zlb, zub,\
           active_set=active_set, Xb=X@beta, rel_tol=1e-8, maxiter=maxiter, verbose=False, delta=1.)
        # new_bound1 = get_dual_cost_given_loss_name(loss_name, X, y, beta1, Xb1, l0, l2, M, ratio,\
        #                                 threshold, zlb, zub, active_set=np.array(range(p)), delta=1.)
        # new_bounds[i,1] = max(dual_cost, new_bound1)
        new_bounds[i,1] = cost1
        zlb[relax_index[i]] = 0      # change zlb, zub back to original
    #avg_incr = mu*np.min(new_bounds-dual_cost,axis=1)+(1-mu)*np.max(new_bounds-dual_cost,axis=1)
    prod_incr = np.maximum(new_bounds[:,1]-dual_cost,epsilon)*np.maximum(new_bounds[:,0]-dual_cost,epsilon)
    idx = np.argmax(prod_incr)
    return relax_index[idx]

def strong_branching_with_scrn_part1(loss_name, X, y, beta, z, cost, old_bounds, l0, l2, M, ratio, threshold, S_diag, ones_index, zeros_index, tol, act_set, mu=1., maxiter=3000):
    p = len(z)
    if act_set is None:
        active_set = np.array(range(p))
    else:
        active_set = np.array(list(act_set))
    relax_index = np.array(list(set(np.where((z>0)&(z<1))[0])-ones_index-zeros_index))
    #print('relax_index',len(relax_index))
    ones_index = set2arr(ones_index)
    zeros_index = set2arr(zeros_index)
    zlb = np.zeros(p)
    zlb[ones_index] = 1
    zub = np.ones(p)
    zub[zeros_index] = 0
    new_bounds = np.full((p,2),cost)
    dual_bounds = old_bounds.copy()
    for i in range(len(relax_index)):
        zub[relax_index[i]] = 0
        beta0,cost0,Xb0,_ = cd(loss_name, X, y, beta, cost, l0, l2, M, ratio, threshold, S_diag, zlb, zub,\
           active_set=active_set, Xb=X@beta, rel_tol=1e-8, maxiter=maxiter, verbose=False, delta=1.)
        new_bound0 = get_dual_cost_given_loss_name(loss_name, X, y, beta0, Xb0, l0, l2, M, ratio,\
                                        threshold, zlb, zub, active_set=np.array(range(p)), delta=1.)
        # new_bounds[relax_index[i],0] = max(old_bounds[relax_index[i],0], new_bound0)
        new_bounds[relax_index[i],0] = cost0
        dual_bounds[relax_index[i],0] = max(old_bounds[relax_index[i],0], new_bound0)
        zub[relax_index[i]] = 1
        zlb[relax_index[i]] = 1
        beta1,cost1,Xb1,_ = cd(loss_name, X, y, beta, cost, l0, l2, M, ratio, threshold, S_diag, zlb, zub,\
           active_set=active_set, Xb=X@beta, rel_tol=1e-8, maxiter=maxiter, verbose=False, delta=1.)
        new_bound1 = get_dual_cost_given_loss_name(loss_name, X, y, beta1, Xb1, l0, l2, M, ratio,\
                                        threshold, zlb, zub, active_set=np.array(range(p)), delta=1.)
        # new_bounds[relax_index[i],1] = max(old_bounds[relax_index[i],0], new_bound1)
        new_bounds[relax_index[i],1] = cost1
        dual_bounds[relax_index[i],1] = max(old_bounds[relax_index[i],1], new_bound1)
        zlb[relax_index[i]] = 0      # change zlb, zub back to original
    return new_bounds, dual_bounds

def strong_branching_with_scrn_part2(z, new_bounds, dual_cost, ones_index, zeros_index, tol, mu=1.):
    relax_index = np.array(list(set(np.where((z>0)&(z<1))[0])-ones_index-zeros_index))
    if len(relax_index)==0:
        return -1
    bounds = new_bounds[relax_index,:]
    avg_incr = mu*np.min(bounds-dual_cost,axis=1)+(1-mu)*np.max(bounds-dual_cost,axis=1)
    idx = np.argmax(avg_incr)
    return relax_index[idx]

def full_strong_branching_with_scrn_part2(z,new_bounds, dual_cost, ones_index, zeros_index, tol, epsilon=1e-6):
    relax_index = np.array(list(set(np.where((z>0)&(z<1))[0])-ones_index-zeros_index))
    if len(relax_index)==0:
        return -1
    bounds = new_bounds[relax_index,:]
    prod_incr = np.maximum(bounds[:,1]-dual_cost,epsilon)*np.maximum(bounds[:,0]-dual_cost,epsilon)
    idx = np.argmax(prod_incr)
    return relax_index[idx]
    
def new_z(node, index):
    new_ones_index = node.ones_index.copy()
    new_ones_index.add(index)
    new_zeros_index = node.zeros_index.copy()
    new_zeros_index.add(index)
    return new_ones_index, new_zeros_index



def branch(current_node, tol, branching_type, use_warm_start, tree_upper_bound, mu=1., branch_maxiter=3000, epsilon=1e-6):
    if branching_type == 'maxfrac':
        branching_variable = \
            max_fraction_branching(current_node.z, current_node.ones_index, current_node.zeros_index, tol)
    elif branching_type == 'maxz':
        branching_variable = \
            max_z_branching(current_node.primal_beta, current_node.ones_index, current_node.zeros_index, tol)
    elif branching_type == 'maxlb':
        branching_variable = \
            max_lb_branching(current_node.fixed_lbs, current_node.ones_index, current_node.zeros_index, tol)
    elif branching_type == 'minlb':
        branching_variable = \
            min_lb_branching(current_node.fixed_lbs, current_node.ones_index, current_node.zeros_index, tol)
    elif branching_type == 'strong':
        branching_variable = \
            strong_branching(current_node.loss_name, current_node.X, current_node.y, \
                             current_node.primal_beta, current_node.z, current_node.primal_value, \
                             current_node.dual_value, current_node.l0, current_node.l2, current_node.M,\
                             current_node.ratio, current_node.threshold, current_node.S_diag, \
                             current_node.ones_index, current_node.zeros_index, tol, None, mu, \
                             branch_maxiter)
    elif branching_type == 'strong_with_scrn':
        new_bounds, dual_bounds = \
            strong_branching_with_scrn_part1(current_node.loss_name,\
                                             current_node.X, current_node.y,\
                                             current_node.primal_beta, current_node.z,\
                                             current_node.primal_value,\
                                             current_node.fixed_lbs, current_node.l0,\
                                             current_node.l2, current_node.M,\
                                             current_node.ratio, current_node.threshold,\
                                             current_node.S_diag, current_node.ones_index,\
                                             current_node.zeros_index, tol, None, mu,\
                                             branch_maxiter)
        current_node.fixed_lbs = dual_bounds
        second_scrn_prune = current_node.L0_screening(tree_upper_bound)
        if not second_scrn_prune:
            branching_variable = \
                strong_branching_with_scrn_part2(current_node.z, new_bounds,\
                                                 current_node.dual_value, \
                                                 current_node.ones_index, \
                                                 current_node.zeros_index, tol, mu)
        else:
            current_node.status = NodeStatus.PrunedAfterScreening
            return None, None, second_scrn_prune
    elif branching_type == 'strong_act':
        branching_variable = \
            strong_branching(current_node.loss_name, current_node.X, current_node.y, \
                             current_node.primal_beta, current_node.z, current_node.primal_value, \
                             current_node.dual_value, current_node.l0, current_node.l2,\
                             current_node.M, current_node.ratio, current_node.threshold, \
                             current_node.S_diag, current_node.ones_index, \
                             current_node.zeros_index, tol, current_node.support, \
                             mu, branch_maxiter)
    elif branching_type == 'strong_act_with_scrn':
        new_bounds, dual_bounds = \
            strong_branching_with_scrn_part1(current_node.loss_name, current_node.X, \
                                             current_node.y, \
                             current_node.primal_beta, current_node.z, current_node.primal_value, \
                             current_node.fixed_lbs, current_node.l0, current_node.l2, current_node.M,\
                             current_node.ratio, current_node.threshold, current_node.S_diag, \
                             current_node.ones_index, current_node.zeros_index, tol, current_node.support, \
                             mu, branch_maxiter)
        current_node.fixed_lbs = dual_bounds
        
        #print('before second screening:',current_node.num_ones['post-screening'],current_node.num_zeros['post-screening'])
        second_scrn_prune = current_node.L0_screening(tree_upper_bound)
        if not second_scrn_prune:
            #print('after second screening:',current_node.num_ones['post-screening'],current_node.num_zeros['post-screening'])
            branching_variable = \
                strong_branching_with_scrn_part2(current_node.z, new_bounds, current_node.dual_value, \
                                                 current_node.ones_index, current_node.zeros_index, tol, mu)
        else:
            current_node.status = NodeStatus.PrunedAfterScreening
            return None, None, second_scrn_prune
    elif branching_type == 'full_strong_act':
        branching_variable = \
            full_strong_branching(current_node.loss_name, current_node.X, current_node.y, \
                                  current_node.primal_beta, current_node.z, \
                                  current_node.primal_value, current_node.dual_value, \
                                  current_node.l0, current_node.l2, current_node.M, \
                                  current_node.ratio, current_node.threshold, \
                                  current_node.S_diag, current_node.ones_index, \
                                  current_node.zeros_index, tol, current_node.support, \
                                  epsilon, branch_maxiter)
    elif branching_type == 'full_strong_act_with_scrn':
        new_bounds, dual_bounds = \
            strong_branching_with_scrn_part1(current_node.loss_name, current_node.X, current_node.y, \
                             current_node.primal_beta, current_node.z, current_node.primal_value, \
                             current_node.fixed_lbs, current_node.l0, current_node.l2, current_node.M,\
                             current_node.ratio, current_node.threshold, current_node.S_diag, \
                             current_node.ones_index, current_node.zeros_index, tol, current_node.support, \
                             mu, branch_maxiter)
        current_node.fixed_lbs = dual_bounds
        
        #print('before second screening:',current_node.num_ones['post-screening'],current_node.num_zeros['post-screening'])
        second_scrn_prune = current_node.L0_screening(tree_upper_bound)
        if not second_scrn_prune:
            #print('after second screening:',current_node.num_ones['post-screening'],current_node.num_zeros['post-screening'])
            branching_variable = \
                full_strong_branching_with_scrn_part2(current_node.z, new_bounds, current_node.dual_value, \
                                                 current_node.ones_index, current_node.zeros_index, tol)
        else:
            current_node.status = NodeStatus.PrunedAfterScreening
            return None, None, second_scrn_prune
    else:
        raise ValueError(f'branching type {branching_type} is not supported')
    if branching_variable == -1:
        if use_warm_start:
            warm_start = dict()
            warm_start['support'] = current_node.support.copy()
            warm_start['beta'] = np.copy(current_node.primal_beta)
        else:
            warm_start = None
        left_node = Node(current_node, current_node.index*2, current_node.ones_index, current_node.zeros_index, warm_start=warm_start)
        right_node = None
    else:
        new_ones_index, new_zeros_index = new_z(current_node, branching_variable)
        if use_warm_start:
            warm_start = dict()
            warm_start['support'] = current_node.support.copy()
            warm_start['beta'] = np.copy(current_node.primal_beta)
        else:
            warm_start = None
        right_node = Node(current_node, current_node.index*2+1, new_ones_index, current_node.zeros_index, warm_start=warm_start)
        if use_warm_start:
            warm_start = dict()
            warm_start['support'] = current_node.support.copy()
            warm_start['beta'] = np.copy(current_node.primal_beta)
        else:
            warm_start = None
        left_node = Node(current_node, current_node.index*2, current_node.ones_index, new_zeros_index, warm_start=warm_start)
    current_node.status = NodeStatus.Branched

    
    return left_node, right_node, 0