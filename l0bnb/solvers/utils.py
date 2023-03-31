import numpy as np
from numba import njit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from sklearn.utils import extmath

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

@njit(cache=True)
def is_integral(z, tol):
    return np.all((z<=tol)|(z>=1-tol))

@njit(cache=True)
def get_ratio_threshold(l0, l2, m):
    ratio = np.sqrt(l0 / l2) if l2 != 0 else np.Inf
    threshold = 2 * np.sqrt(l0 * l2) if ratio <= m else l0 / m + l2 * m
    return ratio, threshold


@njit(cache=True)
def compute_relative_gap(cost1, cost2, which="both", one=True):
    if cost1 == np.inf or cost2 == -np.inf:
        return 1.
    if cost1 == -np.inf or cost2 == np.inf:
        return -1.
    if which == "both":
        benchmark = max(abs(cost1),abs(cost2))
    elif which == "first":
        benchmark = abs(cost1)
    elif which == "second":
        benchmark = abs(cost2)
    if one:
        benchmark = max(benchmark,1)
    return (cost1-cost2)/benchmark


@njit(cache=True)
def trivial_soln(X):
    n,p = X.shape
    return np.zeros(p), np.zeros(n)

@njit(cache=True)
def nb_set2arr(support):
    return np.array(list(support))

def set2arr(support):
    if len(support) == 0:
        return np.array([],dtype=int)
    else:
        return nb_set2arr(support)

@njit(cache=True)
def get_active_set_mask(active_set,p):
    mask = np.full(p, False)
    mask[active_set] = True
    return mask

# @njit(cache=True)
# def active_set_to_support(active_set):
#     if len(active_set) == 0:
#         support = {0}
#         support.clear()
#     else:
#         support = set(active_set)
#     return support


def skl_svd(X):
    return extmath.randomized_svd(X,n_components=1)[1][0]