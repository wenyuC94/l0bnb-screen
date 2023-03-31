import numpy as np
from numba import njit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


@njit
def compute_lower_bounds_for_fixed_z_with_optimal_solution(loss_name, X, y, beta, z, zlb, zub, l0, l2, M, Xb, dual_cost, grad, delta=1.):
    p = X.shape[1]
    bounds = np.full((p,2), dual_cost)
    for i in range(p):
        if zlb[i] == 1 or zub[i] == 0:
            continue
        
        if z[i] > 0:
            u = 2*beta[i]/z[i]
            v = -grad[i]-l2*u
            if M == np.inf:
                D = l0-l2*u**2/4
            else:
                D = l0-l2*u**2/4-M*abs(v)
        else:
            D = l0-grad[i]**2/4/l2 if abs(grad[i])<=2*l2*M else l0+l2*M**2-M*abs(grad[i])
        
        # bounds[i,0] = max(dual_cost, dual_cost-z[i]*D)
        # bounds[i,1] = max(dual_cost, dual_cost+(1-z[i])*D)
        
        bounds[i,0] = max(dual_cost, dual_cost+np.maximum(-D,0))
        bounds[i,1] = max(dual_cost, dual_cost+D)
        
    return bounds

@njit
def compute_lower_bounds_for_fixed_z_with_approximate_solution(loss_name, X, y, beta, z, zlb, zub, l0, l2, M, Xb, dual_cost, grad, loss, delta=1.):
    p = X.shape[1]
    bounds = np.full((p,2), dual_cost)
    new_bound = dual_cost        ## checked empirically, to be checked in theory
    # new_bound = loss-grad@beta
    # for i in range(p):
    #     if zub[i] == 0:
    #         continue
    #     D = l0-grad[i]**2/4/l2 if np.abs(grad[i])<=2*l2*M else l0+l2*M**2-M*np.abs(grad[i])
    #     if zlb[i] == 1:
    #         new_bound += D
    #     else:
    #         new_bound -= np.maximum(-D,0)
    for i in range(p):
        if zlb[i] == 1 or zub[i] == 0:
            continue
        
        D = l0-grad[i]**2/4/l2 if np.abs(grad[i])<=2*l2*M else l0+l2*M**2-M*np.abs(grad[i])
        # bounds[i,0] = max(dual_cost, new_bound-z[i]*D)
        # bounds[i,1] = max(dual_cost, new_bound+(1-z[i])*D)
        
        bounds[i,0] = max(dual_cost, new_bound+np.maximum(-D,0))
        bounds[i,1] = max(dual_cost, new_bound+np.maximum(D,0))
        
    return bounds

@njit
def compute_lower_bounds_for_fixed_z_with_approximate_solution_new(loss_name, X, y, beta, z, zlb, zub, l0, l2, M, Xb, dual_cost, grad, ratio, threshold, delta=1.):
    p = X.shape[1]
    bounds = np.full((p,2), dual_cost)
    a = 2*M*l2 if l2 != 0 else 0
    c = a if ratio <= M else (l0/M+l2*M)
    for i in range(p):
        if zlb[i] == 1 or zub[i] == 0:
            continue
        
        abs_grad = abs(grad[i])
        if abs_grad <= threshold:
            psi_star = 0
        elif abs_grad <= c:
            psi_star = abs_grad**2/(4*l2)-l0
        else:
            psi_star = M*abs_grad-l0-l2*M**2
        if abs_grad <= a:
            phi_1_star = abs_grad**2/(4*l2)-l0
        else:
            phi_1_star = M*abs_grad-l0-l2*M**2
        bounds[i,0] = max(dual_cost, dual_cost+psi_star)
        bounds[i,1] = max(dual_cost, dual_cost+psi_star-phi_1_star)
    return bounds



def L0_screening(bounds, tree_upper_bound):
    fix_to_one = set(np.where(bounds[:,0] > tree_upper_bound)[0])
    fix_to_zero = set(np.where(bounds[:,1] > tree_upper_bound)[0])
    num_fix_to_both = len(fix_to_one.intersection(fix_to_zero))
    return num_fix_to_both, fix_to_one, fix_to_zero

def L0_screening_approx(bounds, tree_upper_bound, gap_tol):
    fix_to_one = set(np.where(bounds[:,0] > tree_upper_bound*(1-gap_tol))[0])
    fix_to_zero = set(np.where(bounds[:,1] > tree_upper_bound*(1-gap_tol))[0])
    num_fix_to_both = len(fix_to_one.intersection(fix_to_zero))
    return num_fix_to_both, fix_to_one, fix_to_zero
    
def relaxation_screening():
    pass