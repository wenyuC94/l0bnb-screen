import numpy as np
from numba import njit

from ...loss_utils import compute_loss, compute_loss_grad


@njit
def get_L0L2_cost(loss_name, X, y, beta, Xb, l0, l2, M, active_set, delta=1.):
    loss = compute_loss(loss_name, y, X, beta, Xb, delta)
    if len(active_set) == 0:
        return loss
    cost = loss + l0*np.sum(np.abs(beta[active_set])>0) + l2*np.sum(beta[active_set]**2)
    return cost

@njit
def get_L2_primal_cost(loss_name, X, y, beta, Xb, l2, M, active_set, delta=1.):
    loss = compute_loss(loss_name, y, X, beta, Xb, delta)
    if len(active_set) == 0:
        return loss
    cost = loss + l2*np.sum(beta[active_set]**2)
    return cost

@njit
def get_L2_dual_cost_given_loss_value(X, y, beta, loss, grad, l2, M, active_set):
    res = loss-beta@grad
    a = 2*M*l2 if l2 != 0 else 0
    pen = 0.
    for i in active_set:
        abs_grad = abs(grad[i])
        if l2 == 0:
            pen += M*abs_grad
        elif abs_grad <= a:
            pen += abs_grad**2/(4*l2)
        else:
            pen += (M*abs_grad-l2*M**2)
    return res-pen

@njit
def get_L2_dual_cost_given_loss_value(loss_name, X, y, beta, loss, grad, l2, M, active_set, delta=1.):
    loss, grad = compute_loss_grad(loss_name, y, X, beta, Xb, delta)
    return get_L2_dual_cost_given_loss_value(X, y, beta, loss, grad, l2, M, active_set)