import numpy as np
from numba import njit
from numba.typed import List


from ..utils import compute_relative_gap, skl_svd
from ..oracle import Q_psi, Q_phi
from ._cost import get_primal_cost
from ...loss_utils import compute_coordinate_grad_Lipschitz

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


@njit
def pg(loss_name, X, y, beta, cost, l0, l2, M, ratio, threshold, L, zlb, zub, active_set, Xb, rel_tol=1e-8, maxiter=3000, verbose=False, delta=1.):
    tol = 1
    old_cost = cost
    curiter = 0
    while tol > rel_tol and curiter < maxiter:
        old_cost = cost
        beta_old = beta.copy()
        if loss_name == 'lstsq':
            grad = (Xb-y)@X
        elif loss_name == 'logistic':
            grad = -(y/(np.exp(y*Xb)+1))@X
        elif loss_name == 'sqhinge':
            grad = -2*(y*np.maximum(1-y*Xb,0))@X
        elif loss_name == 'hbhinge':
            grad = -(y*np.maximum(np.minimum((1-y*Xb)/delta,1),0))@X
        for i in active_set:
            if zub[i] == 0:
                Xb = Xb - beta_old[i]*X[:,i]
                beta[i] = 0
            else:
                beta_tilde = beta_old[i] - grad[i]/L
                if zlb[i] == 0:
                    beta[i] = Q_psi(L/2, -L*beta_tilde, l0, l2, M, ratio, threshold, suff=True)
                else:
                    beta[i] = Q_phi(L/2, -L*beta_tilde, 1, l2, M)
                Xb = Xb + (beta[i] - beta_old[i])*X[:,i]
        cost, loss, _ = get_primal_cost(loss_name, X, y, beta, Xb, l0, l2, M, ratio, threshold, zlb, zub, active_set, delta)
        if verbose:
            print(cost)
        tol = abs(compute_relative_gap(old_cost, cost))
        curiter += 1
    return beta, cost, Xb, loss
