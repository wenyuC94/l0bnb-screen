import numpy as np
from numba import njit
from numba.typed import List

from ..utils import compute_relative_gap
from ._cost import get_L2_primal_cost, get_L0L2_cost
from ..oracle import Q_L2reg, Q_L0L2reg
from ...loss_utils import compute_coordinate_grad_Lipschitz

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)



@njit
def L2_CD_loop(loss_name, X, y, beta, l2, M, S_diag, active_set, Xb, delta=1.):
    beta_old = 0.
    for i in active_set:
        beta_old = beta[i]
        gradi, Li = compute_coordinate_grad_Lipschitz(loss_name, i, y, X, beta, Xb, S_diag, delta)
        beta_tilde = beta_old - gradi/Li
        beta[i] = Q_L2reg(Li/2, -Li*beta_tilde, l2, M)
        Xb = Xb + (beta[i] - beta_old)*X[:,i]
    return beta, Xb


@njit
def L2_CD(loss_name, X, y, beta, cost, l2, M, S_diag, active_set, Xb, rel_tol=1e-8, maxiter=3000, verbose=False, delta=1.):
    tol = 1
    old_cost = cost
    curiter = 0
    while tol > rel_tol and curiter < maxiter:
        old_cost = cost
        beta, Xb = L2_CD_loop(loss_name, X, y, beta, l2, M, S_diag, active_set, Xb, delta)
        cost = get_L2_primal_cost(loss_name, X, y, beta, Xb, l2, M, active_set, delta)
        if verbose:
            print(cost)
        tol = abs(compute_relative_gap(old_cost, cost))
        curiter += 1
    return beta, cost, Xb


@njit
def L0L2_CD_loop(loss_name, X, y, beta, l0, l2, M, S_diag, active_set, Xb, delta=1.):
    beta_old = 0.
    for i in active_set:
        beta_old = beta[i]
        gradi, Li = compute_coordinate_grad_Lipschitz(loss_name, i, y, X, beta, Xb, S_diag, delta)
        beta_tilde = beta_old - gradi/Li
        beta[i] = Q_L0L2reg(Li/2, -Li*beta_tilde, l0, l2, M)
        Xb = Xb + (beta[i] - beta_old)*X[:,i]
    return beta, Xb


@njit
def L0L2_CD(loss_name, X, y, beta, cost, l0, l2, M, S_diag, active_set, Xb, rel_tol=1e-8, maxiter=3000, verbose=False, delta=1.):
    tol = 1
    old_cost = cost
    curiter = 0
    while tol > rel_tol and curiter < maxiter:
        old_cost = cost
        beta, Xb = L0L2_CD_loop(loss_name, X, y, beta, l0, l2, M, S_diag, active_set, Xb, delta)
        cost = get_L0L2_cost(loss_name, X, y, beta, Xb, l0, l2, M, active_set, delta)
        if verbose:
            print(cost)
        tol = abs(compute_relative_gap(old_cost, cost))
        curiter += 1
    return beta, cost, Xb
