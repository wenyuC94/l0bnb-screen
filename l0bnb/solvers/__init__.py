import warnings

from numba.core.errors import NumbaDeprecationWarning, \
    NumbaPendingDeprecationWarning, NumbaPerformanceWarning

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

from . import utils, oracle, relaxation, mosek, screening


# __all__ = ['utils','oracle']