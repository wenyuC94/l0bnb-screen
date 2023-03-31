import numpy as np
from numba import njit
from numba.typed import List

from ..utils import compute_relative_gap
from ..oracle import Q_psi, Q_phi
from ._cost import get_primal_cost
from ...loss_utils import compute_coordinate_grad_Lipschitz

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)



@njit
def cd_loop(loss_name, X, y, beta, l0, l2, M, ratio, threshold, S_diag, zlb, zub, active_set, Xb, delta=1.):
    for i in active_set:
        if zub[i] == 0:
            Xb = Xb - beta[i]*X[:,i]
            beta[i] = 0
        else:
            beta_old = beta[i]
            gradi, Li = compute_coordinate_grad_Lipschitz(loss_name, i, y, X, beta, Xb, S_diag, delta)
            beta_tilde = beta_old - gradi/Li
            if zlb[i] == 0:
                beta[i] = Q_psi(Li/2, -Li*beta_tilde, l0, l2, M, ratio, threshold, suff=True)
            else:
                beta[i] = Q_phi(Li/2, -Li*beta_tilde, 1, l2, M)
            Xb = Xb + (beta[i] - beta_old)*X[:,i]
    return beta, Xb



@njit
def cd(loss_name, X, y, beta, cost, l0, l2, M, ratio, threshold, S_diag, zlb, zub, active_set, Xb, rel_tol=1e-8, maxiter=3000, verbose=False, delta=1.):
    tol = 1
    old_cost = cost
    curiter = 0
    while tol > rel_tol and curiter < maxiter:
        old_cost = cost
        beta, Xb = cd_loop(loss_name, X, y, beta, l0, l2, M, ratio, threshold, S_diag, zlb, zub, active_set, Xb, delta)
        cost, loss, _ = get_primal_cost(loss_name, X, y, beta, Xb, l0, l2, M, ratio, threshold, zlb, zub, active_set, delta)
        if verbose:
            print(cost)
        tol = abs(compute_relative_gap(old_cost, cost))
        curiter += 1
    if verbose:
        if curiter == maxiter:
            print('Maximum CD check iterations reached')
    return beta, cost, Xb, loss

