import numpy as np
from numba import njit

from ..utils import get_ratio_threshold
from ...loss_utils import compute_loss, compute_loss_grad

@njit
def get_primal_cost(loss_name, X, y, beta, Xb, l0, l2, M, ratio, threshold, zlb, zub, active_set, delta=1.):
    loss = compute_loss(loss_name, y, X, beta, Xb, delta)
    
    if len(active_set) == 0:
        return loss, loss, np.zeros_like(zlb)
    
    abs_beta = np.abs(beta[active_set])
    zlb_act = zlb[active_set]
    zub_act = zub[active_set]
    
    z = np.zeros_like(zlb)
    z_act = abs_beta/M if ratio > M else abs_beta/ratio
    z_act = np.minimum(np.maximum(zlb_act, z_act), zub_act)
    z[active_set] = z_act
    s_act = np.where(zlb_act==1, abs_beta**2, \
                 abs_beta*M if ratio > M else np.maximum(abs_beta*ratio, abs_beta**2))
    
    return loss+l0*np.sum(z_act[z_act>0])+l2*np.sum(s_act[s_act>0]), loss, z

@njit
def get_dual_cost_given_loss_value(X, y, beta, loss, grad, l0, l2, M, ratio, threshold, zlb, zub, active_set):
    res = loss-beta@grad
    a = 2*M*l2 if l2 != 0 else 0
    c = a if ratio <= M else (l0/M+l2*M)
    pen = 0.
    abs_grad = 0.
    for i in active_set:
        abs_grad = abs(grad[i])
        if zub[i] == 0:
            continue
        elif zlb[i] == 1:
            if l2 == 0:
                pen += (M*abs_grad-l0)
            elif abs_grad <= a:
                pen += (abs_grad**2/(4*l2)-l0)
            else:
                pen += (M*abs_grad-l0-l2*M**2)
        else:
            if abs_grad <= threshold:
                continue
            elif abs_grad <= c:
                pen += (abs_grad**2/(4*l2)-l0)
            else:
                pen += (M*abs_grad-l0-l2*M**2)
    return res - pen
    
@njit(cache=True)
def get_dual_cost_given_loss_name(loss_name, X, y, beta, Xb, l0, l2, M, ratio, threshold, zlb, zub, active_set, delta=1.):
    loss, grad = compute_loss_grad(loss_name, y, X, beta, Xb, delta)
    return get_dual_cost_given_loss_value(X, y, beta, loss, grad, l0, l2, M, ratio, threshold, zlb, zub, active_set)

