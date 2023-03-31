from copy import deepcopy

import numpy as np

from ..solvers.relaxation import relax_ASCD_solve, relax_ASPG_solve
from ..solvers.mosek import relax_mosek_solve, relax_ASmosek_solve
from ..solvers.heuristics import heuristic_solve
from ..solvers.utils import get_ratio_threshold
from ..solvers.screening import L0_screening, L0_screening_approx

class NodeStatus:
    Initialized = 0
    PrunedWithoutSolving = 1
    RelaxationScreened = 2
    LowerSolved = 3
    UpperSolved = 4
    Integral = 5
    PrunedBeforeScreening = 6
    L0Screened = 7
    PrunedAfterScreening = 8
    Branching = 9
    Branched = 10
    
    @staticmethod
    def IsPruned(status):
        return status in {NodeStatus.PrunedWithoutSolving, NodeStatus.PrunedBeforeScreening, NodeStatus.PrunedAfterScreening}
    
    @staticmethod
    def HasLowerSolved(status):
        return status >= NodeStatus.LowerSolved
    
    @staticmethod
    def HasUpperSolved(status):
        return status >= NodeStatus.UpperSolved
    
    @staticmethod
    def HasL0Screened(status):
        return status >= NodeStatus.L0Screened
    
NodeStatusLookupDict = {getattr(NodeStatus,status):status for status in dict(vars(NodeStatus)) if '__' not in status}

class Node:
    def __init__(self, parent, index, ones_index: set, zeros_index: set, **kwargs):
        """
        Initialize a Node

        Parameters
        ----------
        parent: Node or None
            the parent Node
        ones_index: np.array
            p x 1 array representing the lower bound of the integer variables z
        zeros_index: np.array
            p x 1 array representing the upper bound of the integer variables z

        Other Parameters
        ----------------
        X: np.array
            The data matrix (n x p). If not specified the data will be inherited
            from the parent node
        y: np.array
            The response (n x 1). If not specified the data will be inherited
            from the parent node 
        S_diag: np.array
            The diagonal of X.T@X. If not specified the data will
            be inherited from the parent node
        l0: float
            The zeroth norm coefficient. If not specified the data will
            be inherited from the parent node
        l2: float
            The second norm coefficient. If not specified the data will
            be inherited from the parent node
        M: float
            The bound for the features (\beta). If not specified the data will
            be inherited from the parent node
        """
        
        self.X = kwargs.get('X', parent.X if parent else None)
        self.y = kwargs.get('y', parent.y if parent else None)
        self.p = self.X.shape[1]
        self.S_diag = kwargs.get('S_diag',
                                  parent.S_diag if parent else None)
        self.L = kwargs.get('L', parent.L if parent else None)
        
        self.l0 = kwargs.get('l0', parent.l0 if parent else None)
        self.l2 = kwargs.get('l2', parent.l2 if parent else None)
        self.M = kwargs.get('M', parent.M if parent else None)
        self.loss_name = kwargs.get('loss_name', parent.loss_name if parent else None)
        self.delta = kwargs.get('delta', parent.delta if parent else None)
        
        self.ratio = kwargs.get('ratio', parent.ratio if parent else None)
        self.threshold = kwargs.get('threshold', parent.threshold if parent else None)
        if self.ratio is None or self.threshold is None:
            self.ratio, self.threshold = get_ratio_threshold(self.l0, self.l2, self.M)
        
        self.parent_dual = parent.dual_value if parent else None
        self.parent_primal = parent.primal_value if parent else None

        self.warm_start = kwargs.get('warm_start', None)
        
        self.index = index
        self.level = parent.level + 1 if parent else 0

        self.ones_index = ones_index
        self.zeros_index = zeros_index
        self.z = None

        self.upper_bound = None
        self.primal_value = None
        self.dual_value = None

        self.support = None
        self.upper_beta = None
        self.primal_beta = None
        
        self.upper_active_set_mask = None
        
        self.num_ones = {'pre-screening':len(ones_index)}
        self.num_zeros = {'pre-screening':len(zeros_index)}
        self.num_free = {'pre-screening':self.p-len(ones_index)-len(zeros_index)}
        self.num_both = dict()
        
        self.status = NodeStatus.Initialized
        
    def lower_solve(self, solver, active_set_coeff=1., int_tol=1e-4, rel_tol=1e-4, msk_tol=1e-8, tree_upper_bound=np.inf, mio_gap=1e-2, ascd_heuristics=True, cd_max_itr=100, kkt_max_itr=100, compute_fixed_lbs=False, fixed_lb_method='exact', verbose=False, sp_ratio=0.2):
        
        if solver == "ASCD":  
            sol = relax_ASCD_solve(self.loss_name, self.X, self.y, self.l0, self.l2, self.M, self.ones_index, self.zeros_index, self.upper_active_set_mask, self.delta,
                ratio=self.ratio, threshold=self.threshold, S_diag=self.S_diag, warm_start=self.warm_start, 
                rel_tol=rel_tol, int_tol=int_tol, tree_upper_bound=tree_upper_bound, mio_gap=mio_gap, check_if_integral=True, ascd_heuristics=ascd_heuristics, 
                cd_max_itr=cd_max_itr, kkt_max_itr=kkt_max_itr, compute_fixed_lbs=compute_fixed_lbs, fixed_lb_method=fixed_lb_method, verbose=verbose, sp_ratio=sp_ratio)
        
        elif solver == "ASPG":  
            sol = relax_ASPG_solve(self.loss_name, self.X, self.y, self.l0, self.l2, self.M, self.ones_index, self.zeros_index, self.upper_active_set_mask, self.delta,
                ratio=self.ratio, threshold=self.threshold, L=self.L, warm_start=self.warm_start, 
                rel_tol=rel_tol, int_tol=int_tol, tree_upper_bound=tree_upper_bound, mio_gap=mio_gap, check_if_integral=True, ascd_heuristics=ascd_heuristics, 
                cd_max_itr=cd_max_itr, kkt_max_itr=kkt_max_itr, compute_fixed_lbs=compute_fixed_lbs, fixed_lb_method=fixed_lb_method, verbose=verbose)

        elif solver == 'ASMosek':
            sol = relax_ASmosek_solve(self.loss_name, self.X, self.y, self.l0, self.l2, self.M, self.ones_index, self.zeros_index, upper_active_set_mask = self.upper_active_set_mask, active_set_coeff=active_set_coeff, delta=self.delta, ratio=self.ratio, threshold=self.threshold, warm_start=self.warm_start, int_tol=int_tol, msk_tol=msk_tol, check_if_integral=True, kkt_max_itr=kkt_max_itr, compute_fixed_lbs=compute_fixed_lbs, fixed_lb_method=fixed_lb_method, verbose=verbose)
        
        elif solver == 'Mosek':
            sol = relax_mosek_solve(self.loss_name, self.X, self.y, self.l0, self.l2, self.M, self.ones_index, self.zeros_index, upper_active_set_mask = self.upper_active_set_mask, delta=self.delta, threshold=self.threshold, compute_fixed_lbs=compute_fixed_lbs, fixed_lb_method=fixed_lb_method)
            
        self.primal_value = sol.primal_value
        self.dual_value = sol.dual_value
        self.primal_beta = sol.beta
        self.z = sol.z
        self.support = sol.support
        self.Xb = sol.Xb
        self.fixed_lbs = sol.lower_bounds
        
        self.status = NodeStatus.LowerSolved
        return self.primal_value, self.dual_value

    
    
    def upper_solve(self, solver, **kwargs):
        upper_beta, upper_bound, _, _ = heuristic_solve(self.loss_name, self.X, self.y, self.l0, self.l2, self.M, self.delta, beta=np.copy(self.primal_beta), z=self.z, S_diag=self.S_diag, timing=False, maxtime=np.inf, **kwargs)
        
        self.upper_bound = upper_bound
        self.upper_beta = upper_beta
        
        self.status = NodeStatus.UpperSolved
        return upper_bound
    
    def L0_screening(self, tree_upper_bound):
        num_fix_to_both, fix_to_one, fix_to_zero = L0_screening(self.fixed_lbs, tree_upper_bound)
        self.status = NodeStatus.L0Screened
        # a=1e-5
        # print('sol.z [1-a,1]:', sum(self.z>=1-a))
        # if len(fix_to_zero)>0:
        #     print('screened to zero [1-a,1]:', sum(self.z[list(fix_to_zero)]>=1-a))
        # if len(fix_to_one)>0:
        #     print('screened to one [1-a,1]:', sum(self.z[list(fix_to_one)]>=1-a))
        self.ones_index = self.ones_index | fix_to_one
        self.zeros_index = self.zeros_index | fix_to_zero
        self.num_ones['post-screening'] = len(self.ones_index)
        self.num_zeros['post-screening'] = len(self.zeros_index)
        self.num_free['post-screening'] = self.p - len(self.ones_index) - len(self.zeros_index) + num_fix_to_both
        self.num_both['post-screening'] = num_fix_to_both
        self.status = NodeStatus.Branching if num_fix_to_both==0 else NodeStatus.PrunedAfterScreening
        return num_fix_to_both > 0
    
    def L0_screening_approx(self, tree_upper_bound, gap_tol):
        num_fix_to_both, fix_to_one, fix_to_zero = L0_screening_approx(self.fixed_lbs, \
                                                                       tree_upper_bound, gap_tol)
        self.status = NodeStatus.L0Screened
        self.ones_index = self.ones_index | fix_to_one
        self.zeros_index = self.zeros_index | fix_to_zero
        self.num_ones['post-screening'] = len(self.ones_index)
        self.num_zeros['post-screening'] = len(self.zeros_index)
        self.num_free['post-screening'] = self.p - len(self.ones_index) - len(self.zeros_index) + num_fix_to_both
        self.num_both['post-screening'] = num_fix_to_both
        self.status = NodeStatus.Branching if num_fix_to_both==0 else NodeStatus.PrunedAfterScreening
        return num_fix_to_both > 0
    
    def relaxation_screening(self):
        self.status = NodeStatus.RelaxationScreened
        pass
    
    @property
    def z_summary(self):
        return {key:getattr(self,key) for key in ['num_ones','num_zeros','num_free','num_both']}
    
    def __str__(self):
        return f'level: {self.level}, index: {self.index}, lower cost: {self.primal_value}, ' \
            f'upper cost: {self.upper_bound}'

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        if self.level == other.level:
            if self.primal_value is None and other.primal_value:
                return True
            if other.primal_value is None and self.primal_value:
                return False
            elif not self.primal_value and not other.primal_value:
                return self.parent_primal > \
                       other.parent_cost
            return self.primal_value > other.lower_bound_value
        return self.level < other.level
