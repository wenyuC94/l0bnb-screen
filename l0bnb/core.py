import time
import queue
import sys
from collections import namedtuple
from types import SimpleNamespace

import numpy as np

from .node import Node, NodeStatus
from .solvers.heuristics import heuristic_solve
from .tree_utils import branch
from .solvers.utils import compute_relative_gap, get_ratio_threshold, is_integral
from .data_utils import preprocess
from .solvers.relaxation import _cost

class BNBTree:
    def __init__(self, loss_name, X, y, delta=1.):
        """
        Initiate a BnB Tree to solve the pseudolikelihood problem with
        l0l2 regularization
        Parameters
        ----------
        loss_name: str, 
            'lstsq', 'logistic', 'sqhinge', or 'hbhinge': the name of loss function used
        X: np.array
            n x p numpy array
        y: np.array
            numpy array of length n
        delta: float, default 1.0
            parameter of huberized hinge
        
        assume_centered: bool, default=False
            If True, data will not be centered before computation. If False (default), data will be centered before computation.
        """
        self.loss_name = loss_name
        self.delta = delta
        self.n, self.p, self.X, self.y, self.S_diag = preprocess(X,y)
        

        

        self.bfs_queue = None
        self.dfs_queue = None

        self.levels = {}

        self.root = None
    
    def init_params(self, **kwargs):
        self.params = SimpleNamespace()
        
        self.BnBParams = SimpleNamespace()
        self.BnBParams.int_tol = kwargs.get('int_tol', 1e-4)
        self.BnBParams.gap_tol = kwargs.get('gap_tol', 1e-2)
        self.BnBParams.branching = kwargs.get('branching', 'maxfrac')
        assert self.BnBParams.branching in {'maxfrac','maxz','maxlb','minlb',\
                                            'strong','strong_with_scrn',\
                                            'strong_act','strong_act_with_scrn',\
                                           'full_strong_act','full_strong_act_with_scrn'}
        self.BnBParams.mu = kwargs.get('mu', 0)
        self.BnBParams.branch_maxiter = kwargs.get('branch_maxiter', 2)
        self.BnBParams.max_depth = kwargs.get('max_depth',np.inf)
        self.BnBParams.number_of_dfs_levels = kwargs.get('number_of_dfs_levels',0)
        self.BnBParams.time_limit = kwargs.get('time_limit', 3600)
        self.BnBParams.verbose = kwargs.get('verbose', False)
        self.BnBParams.default_lb = kwargs.get('default_lb', -sys.maxsize)
        self.BnBParams.upper_solver = kwargs.get('upper_solver','L0L2_ASCD')
        assert self.BnBParams.upper_solver in {'L0L2_ASCD','Mosek'}
        self.BnBParams.lower_solver = kwargs.get('lower_solver','ASCD')
        assert self.BnBParams.lower_solver  in {'ASCD','ASPG','ASMosek','Mosek'}
        self.BnBParams.perform_l0_scrn = kwargs.get('perform_l0_scrn', True)
        self.BnBParams.scrn_type = kwargs.get('scrn_type', 'node')
        assert self.BnBParams.scrn_type in {'node', 'root'}
        self.BnBParams.scrn_heuristics = kwargs.get('scrn_heuristics',True)
        self.BnBParams.prune_heuristics = kwargs.get('prune_heuristics',False)
        self.BnBParams.perform_rlx_scrn = kwargs.get('perform_rlx_scrn', False)
        self.BnBParams.fixed_lb_method = kwargs.get('fixed_lb_method','approx')
        assert self.BnBParams.fixed_lb_method in {'exact','approx','approx-new','approx-combined'}
        self.BnBParams.use_warm_start = kwargs.get('use_warm_start', True)
        
        self.__dict__.update(**self.BnBParams.__dict__)
        
        
        self.ScreeningParams = SimpleNamespace()
        self.ScreeningParams.rlx_scrn_method = None
        
        self.UpperSolverParams = SimpleNamespace()
        self.UpperSolverParams.upper_support_heuristic = kwargs.get('upper_support_heuristic', 'nonzeros')
        self.UpperSolverParams.cd_max_itr = kwargs.get('upper_cd_max_itr', 100)
        self.UpperSolverParams.kkt_max_itr = kwargs.get('upper_kkt_max_itr', 100)
        self.UpperSolverParams.rel_tol = kwargs.get('upper_rel_tol', 1e-4)
        self.UpperSolverParams.cd_tol = kwargs.get('upper_cd_tol', 1e-4)
        self.UpperSolverParams.verbose = False
        
        self.LowerSolverParams = SimpleNamespace()
        self.LowerSolverParams.cd_max_itr = kwargs.get('lower_cd_max_itr', 100)
        self.LowerSolverParams.kkt_max_itr = kwargs.get('lower_kkt_max_itr', 100)
        self.LowerSolverParams.rel_tol = kwargs.get('lower_rel_tol', 1e-4)
        if self.branching == 'strong' or self.perform_l0_scrn:
            self.LowerSolverParams.compute_fixed_lbs = True
            self.LowerSolverParams.fixed_lb_method = self.fixed_lb_method
        else:
            self.LowerSolverParams.compute_fixed_lbs = False
            self.LowerSolverParams.fixed_lb_method = None
        self.LowerSolverParams.active_set_coeff = kwargs.get('active_set_coeff', 1.)
        self.LowerSolverParams.msk_tol = kwargs.get('msk_tol', 1e-8)
        self.LowerSolverParams.ascd_heuristics = kwargs.get('ascd_heuristics', False)
        self.LowerSolverParams.verbose = kwargs.get('lower_verbose', False)
        self.LowerSolverParams.sp_ratio = kwargs.get('sp_ratio', 0.2)
        self.best_gap = np.inf

        
    def solve(self, l0, l2, M, beta=None,**kwargs):
        """
        Solve the pseudolikelihood problem with l0l2 regularization
        Parameters
        ----------
        ****** Problem setup ******
        l0: float
            The zeroth norm coefficient
        l2: float
            The second norm coefficient
        M: float
            features bound (big M)
        beta: np.array, optional
            p x 1 array representing a warm start of beta
        
        ****** BnBTree params ****** 
        gap_tol: float, optional
            the relative gap between the upper and lower bound after which the
            algorithm will be terminated. Default 1e-2
        int_tol: float, optional
            the tolerance of z considered as integer. Default 1e-4            
        branching: str, optional
            'maxfrac' or 'strong'. Default 'maxfrac'
        number_of_dfs_levels: int, optional
            number of levels to solve as dfs. Default is 0
        time_limit: float, optional
            The time (in seconds) after which the solver terminates.
            Default is 3600
        verbose: int, optional
            print progress. Default False
        upper_solver: str, optional
            'L2CD', 'L2CDApprox', 'L0L2_CD', 'L0L2_ASCD', 'L0L2_CDPSI' or 'L0L2_ASCDPSI'. Default 'L0L2_ASCD'    
        lower_solver: str, optional
            'ASCD'. Default 'ASCD'
        perform_l0_scrn: bool, optional
            whether to perform l0 screening
        perform_rlx_scrn: bool, optional
            whether to perform relaxation screening
        fixed_lb_method: str, optional
            the method to compute lower bounds with fixed zs
        
        ****** Upper solver params ******
        upper_support_heuristic: str, optional
            'nonzeros', 'rounding', or 'all'. Default 'nonzeros'. 
            The selection method for the support over which the upper solver solves
            the upper problem. The support selection is based on the current lower solution. 
        upper_cd_max_itr: int, optional
            The cd max iterations. Default is 1000
        upper_kkt_max_itr: int, optional
            The kkt check max iterations. Default is 100
        upper_rel_tol: float, optional
            The optimization tolerance between two kkt iters. Default is 1e-4
        upper_cd_tol: float, optional
            The optimization tolerance between two cd iters. Default is 1e-4
        
        ****** Lower solver params ******
        lower_cd_max_itr: int, optional
            The cd max iterations. Default is 1000
        lower_kkt_max_itr: int, optional
            The kkt check max iterations. Default is 100
        lower_rel_tol: float, optional
            The optimization tolerance between two kkt iters. Default is 1e-4
        
        Returns
        -------
        tuple
            A numed tuple with fields 'cost beta sol_time lower_bound gap'
        """
        l0 = float(l0)
        l2 = float(l2)
        M = float(M)
        self.init_params(**kwargs)
        
        st = time.time()
        ratio, threshold = get_ratio_threshold(l0,l2,M)
        
        ## warm start
        upper_bound, upper_beta, support = self._warm_start(beta, l0, l2, M)
            
        if self.verbose:
            print(f"initializing took {time.time() - st} seconds")

        # root node
        if upper_beta is not None:
            warm_start = dict()
            warm_start['support'] = support
            warm_start['beta'] = upper_beta
            self.root = Node(None, 1, set([]), set([]), X=self.X, y=self.y,\
                             loss_name = self.loss_name, delta=self.delta,\
                             S_diag=self.S_diag, l0=l0, l2=l2, M=M, ratio=ratio,\
                             threshold=threshold, warm_start=warm_start)
        else:
            self.root = Node(None, 1, set([]), set([]), X=self.X, y=self.y,\
                             loss_name = self.loss_name, delta=self.delta,\
                             S_diag=self.S_diag, l0=l0, l2=l2, M=M, ratio=ratio,\
                             threshold=threshold)
        self.bfs_queue = queue.Queue()
        self.dfs_queue = queue.LifoQueue()
        self.bfs_queue.put(self.root)

        # lower and upper bounds initialization
        primal_values, dual_values = {}, {}  # level: bound_value
        self.levels = {0: 1}
        self.node_status = {self.root.index: self.root.status}
        self.node_z_summary = {self.root.index: self.root.z_summary}
        self.node_summary = {}
        self.level_summary = {}
        min_open_level = 0

        max_lower_bound_value = self.default_lb
        best_gap = self.gap_tol + 1

        if self.verbose:
            print(f'{self.number_of_dfs_levels} levels of depth used')
        
        curr_level = 0
        not_all_prune = False
        while (self.bfs_queue.qsize() > 0 or self.dfs_queue.qsize() > 0) and \
                (time.time() - st < self.time_limit):

            # get current node
            if self.dfs_queue.qsize() > 0:
                curr_node = self.dfs_queue.get()
            else:
                curr_node = self.bfs_queue.get()
            
            # prune?
            if curr_node.parent_dual and ((self.prune_heuristics and upper_bound*(1-self.gap_tol) <= curr_node.parent_dual) or (upper_bound <= curr_node.parent_dual)):
                
                self.levels[curr_node.level] -= 1
                curr_node.status = NodeStatus.PrunedWithoutSolving
                self.node_status[curr_node.index] = curr_node.status
                
                if self.levels[min_open_level] == 0 and not_all_prune:
                    del self.levels[min_open_level]
                    max_lower_bound_value = max(max([j for i, j in dual_values.items()
                                             if i <= min_open_level]), self.default_lb)
                    best_gap = \
                    (upper_bound - max_lower_bound_value) / abs(upper_bound)
                    self.node_summary[curr_node.index] = {'total_time':time.time()-st,'best_gap':best_gap}
                    self.best_gap =  best_gap 
                    level_indexes = [index for index in self.node_status if (index>=2**curr_level and index<2**(curr_level+1))]
                    nodes_solved = sum([NodeStatus.HasLowerSolved(self.node_status[index]) for index in level_indexes])
                    self.level_summary[curr_level] = {'total_time':time.time()-st,'best_gap':best_gap,\
                                                  'n_nodes':nodes_solved}
                    if self.verbose:
                        print(f'l: {min_open_level}, (d: {max_lower_bound_value}, '
                          f'p: {primal_values.get(min_open_level,0)}), '
                          f'u: {upper_bound}, g: {best_gap}, '
                          f't: {time.time() - st} s, '
                          f'n: {nodes_solved}')
                        print("111")
                    curr_level += 1
                    min_open_level += 1
                    not_all_prune = False
                
                continue
                
            not_all_prune = True
            rel_gap_tol = -1
            if best_gap <= 20 * self.gap_tol or \
                    time.time() - st > self.time_limit / 4:
                rel_gap_tol = 0
            if best_gap <= 10 * self.gap_tol or \
                    time.time() - st > 3 * self.time_limit / 4:
                rel_gap_tol = 1
                
            
            # calculate primal and dual values
            # relaxation screening
            if self.perform_rlx_scrn:
                pass # CALL relaxation screening
                self.node_status[curr_node.index] = curr_node.status
            
            #print(curr_node.index)
            curr_primal, curr_dual = curr_node.lower_solve(self.lower_solver,\
                                                           int_tol=self.int_tol,\
                                                        tree_upper_bound=upper_bound,\
                                                           mio_gap=rel_gap_tol,\
                                                           **self.LowerSolverParams.__dict__)
            #cost000=_cost.get_primal_cost(self.loss_name, self.X, self.y, curr_node.primal_beta, self.X@curr_node.primal_beta, l0, l2, M, ratio, threshold, np.zeros(self.p), np.ones(self.p), np.arange(self.p), delta=self.delta)
            #print(curr_primal,curr_dual,cost000)
            #print(curr_node.primal_beta)
            primal_values[curr_node.level] = \
                min(curr_primal, primal_values.get(curr_node.level, sys.maxsize))
            dual_values[curr_node.level] = \
                min(curr_dual, dual_values.get(curr_node.level, sys.maxsize))
            self.levels[curr_node.level] -= 1
            
            self.node_status[curr_node.index] = curr_node.status
            
            
            # compute feasible integral solution
            
            curr_upper_bound = curr_node.upper_solve(self.upper_solver, **self.UpperSolverParams.__dict__)
            
            self.node_status[curr_node.index] = curr_node.status
            
            # update upper bound
            
            if curr_upper_bound < upper_bound:
                upper_bound = curr_upper_bound
                upper_beta = curr_node.upper_beta
                support = curr_node.support
                best_gap = \
                    (upper_bound - max_lower_bound_value) / abs(upper_bound)
                self.best_gap =  best_gap 

            self.node_summary[curr_node.index] = {'total_time':time.time()-st,'best_gap':best_gap}
            
            # update gap?
            if self.levels[min_open_level] == 0:
                del self.levels[min_open_level]
                max_lower_bound_value = max(max([j for i, j in dual_values.items()
                                             if i <= min_open_level]), self.default_lb)
                best_gap = \
                    (upper_bound - max_lower_bound_value) / abs(upper_bound)
                self.node_summary[curr_node.index]['best_gap'] = best_gap
                self.best_gap =  best_gap   
                level_indexes = [index for index in self.node_status if (index>=2**curr_level and index<2**(curr_level+1))]
                nodes_solved = sum([NodeStatus.HasLowerSolved(self.node_status[index]) for index in level_indexes])
                self.level_summary[curr_level] = {'total_time':time.time()-st,'best_gap':best_gap,\
                                                  'n_nodes':nodes_solved}
                if self.verbose:
                    print(f'l: {min_open_level}, (d: {max_lower_bound_value}, '
                          f'p: {primal_values[min_open_level]}), '
                          f'u: {upper_bound}, g: {best_gap}, '
                          f't: {time.time() - st} s, '
                          f'n: {nodes_solved}')
                      
                curr_level += 1
                    
                min_open_level += 1
                not_all_prune = False
                
                
            # arrived at a solution?
            if best_gap <= self.gap_tol or min_open_level > self.max_depth-1:
                if self.perform_l0_scrn and self.scrn_type == 'node':
                    if self.scrn_heuristics:
                        prune = curr_node.L0_screening_approx(tree_upper_bound=\
                                                                      upper_bound,\
                                                                      gap_tol=self.gap_tol)
                    else:
                        prune = curr_node.L0_screening(tree_upper_bound=upper_bound)
                    self.node_z_summary[curr_node.index] = curr_node.z_summary
                    self.node_status[curr_node.index] = curr_node.status
                if best_gap <= self.gap_tol:
                    print('exit because best gap < tolerance')
                elif min_open_level > self.max_depth-1:
                    print('exit because max depth achieved')
                return self._package_solution(upper_beta, upper_bound,
                                              max_lower_bound_value, best_gap, time.time() - st)

            # integral solution?
            if is_integral(curr_node.z, self.int_tol):
                curr_upper_bound = curr_primal
                if curr_upper_bound < upper_bound:
                    upper_bound = curr_upper_bound
                    upper_beta = curr_node.upper_beta
                    support = curr_node.support
                    if self.verbose:
                        print('integral:', curr_node)
                best_gap = \
                    (upper_bound - max_lower_bound_value) / abs(upper_bound)
                self.best_gap =  best_gap 
                curr_node.status = NodeStatus.Integral
                self.node_status[curr_node.index] = curr_node.status
            
            # branch?
            elif curr_dual < upper_bound:
                prune = False
                if self.perform_l0_scrn:
                    #print('before first screening:',current_node.num_ones['pre-screening'],current_node.num_zeros['pre-screening'])
                    if curr_node.index > 1:
                        if self.scrn_type == 'node':
                            if self.scrn_heuristics:
                                prune = curr_node.L0_screening_approx(tree_upper_bound=\
                                                                      upper_bound,\
                                                                      gap_tol=self.gap_tol)
                            else:
                                prune = curr_node.L0_screening(tree_upper_bound=upper_bound)
                    else:
                        if self.scrn_heuristics:
                            prune = curr_node.L0_screening_approx(tree_upper_bound=\
                                                                      upper_bound,\
                                                                      gap_tol=self.gap_tol)
                        else:
                            prune = curr_node.L0_screening(tree_upper_bound=upper_bound)
                    #print('after first screening:',current_node.num_ones['post-screening'],current_node.num_zeros['post-screening'])
                    self.node_z_summary[curr_node.index] = curr_node.z_summary
                    if prune:
                        self.node_status[curr_node.index] = curr_node.status
                if not prune:
                    # start branching
                    left_node, right_node, second_scrn_prune = branch(curr_node,self.int_tol,self.branching,\
                                                   self.use_warm_start,upper_bound,self.mu,\
                                                                      self.branch_maxiter)
                    self.node_status[curr_node.index] = curr_node.status
                    if self.BnBParams.branching == 'strong_act_with_scrn' or self.BnBParams.branching == \
                    'full_strong_act_with_scrn':
                        self.node_z_summary[curr_node.index]['num_ones']['post-second-screening'] = \
                        curr_node.num_ones['post-screening']
                        self.node_z_summary[curr_node.index]['num_zeros']['post-second-screening'] = \
                        curr_node.num_zeros['post-screening']
                        self.node_z_summary[curr_node.index]['num_free']['post-second-screening'] = \
                        curr_node.num_free['post-screening']
                        # self.node_z_summary[curr_node.index]['num_both']['post-second-screening'] = \
                        # curr_node.num_both['post-screening']
                    if not second_scrn_prune:
                        self.node_status[left_node.index] = left_node.status
                        self.node_z_summary[left_node.index] = left_node.z_summary
                        if right_node is not None:
                            self.node_status[right_node.index] = right_node.status
                            self.node_z_summary[right_node.index] = right_node.z_summary
                            self.levels[curr_node.level + 1] = \
                                self.levels.get(curr_node.level + 1, 0) + 2
                        else:
                            self.levels[curr_node.level + 1] = \
                                self.levels.get(curr_node.level + 1, 0) + 1                
                        if curr_node.level < min_open_level + self.number_of_dfs_levels:
                            if right_node is not None:
                                self.dfs_queue.put(right_node)
                            self.dfs_queue.put(left_node)
                        else:
                            if right_node is not None:
                                self.bfs_queue.put(right_node)
                            self.bfs_queue.put(left_node)
                    
            # prune because curr_dual > upper_bound
            else:
                curr_node.status = NodeStatus.PrunedBeforeScreening
                self.node_status[curr_node.index] = curr_node.status
        if time.time() - st < self.time_limit:
            print('exit because all nodes are visited')
        else:
            print('exit because time out')
        return self._package_solution(upper_beta, upper_bound, max_lower_bound_value,
                                      best_gap,time.time() - st)

    @staticmethod
    def _package_solution(upper_beta, upper_bound, lower_bound, gap, sol_time):
        _sol_str = 'cost beta sol_time lower_bound gap'
        Solution = namedtuple('Solution', _sol_str)
        return Solution(cost=upper_bound, beta=upper_beta, gap=gap,
                        lower_bound=lower_bound, sol_time=sol_time)

    def _warm_start(self, beta, l0, l2, M):
        if beta is None:
            return sys.maxsize, None, None
        else:
            if self.verbose:
                print("used a warm start")
            upper_beta, upper_bound, _, support = heuristic_solve(self.loss_name, self.X, self.y, l0, l2, M, delta=self.delta, solver=self.upper_solver, beta=beta, S_diag=self.S_diag, **self.UpperSolverParams.__dict__)
            return upper_bound, upper_beta, support
        
    def l0screen_summary(self):
        summary = dict()
        #max_level = int(np.max(list(self.level_summary)))
        if self.scrn_type == 'root':
            max_level = 0
        else:
            max_level = int(np.max(list(self.level_summary)))
        
        for level in range(max_level+1):
            level_indexes = [index for index in self.node_status if (index>=2**level and index<2**(level+1))]
            total_ones_screened, total_zeros_screened = 0, 0
            total_ones_screened_both, total_zeros_screened_both = 0, 0
            if self.perform_l0_scrn:
                for index in level_indexes:
                    if NodeStatus.HasL0Screened(self.node_status[index]):
                        total_ones_screened += (self.node_z_summary[index]['num_ones']['post-screening']- self.node_z_summary[index]['num_ones']['pre-screening'])
                        total_zeros_screened += (self.node_z_summary[index]['num_zeros']['post-screening']- self.node_z_summary[index]['num_zeros']['pre-screening'])
                        if self.BnBParams.branching == 'strong_act_with_scrn' or self.BnBParams.branching == \
                        'full_strong_act_with_scrn':
                            try:
                                total_ones_screened_both += (self.node_z_summary[index]['num_ones']['post-second-screening']- self.node_z_summary[index]['num_ones']['pre-screening'])
                                total_zeros_screened_both += (self.node_z_summary[index]['num_zeros']['post-second-screening']- self.node_z_summary[index]['num_zeros']['pre-screening'])
                            except:   # node pruned after first screening
                                total_ones_screened_both += (self.node_z_summary[index]['num_ones']['post-screening']- self.node_z_summary[index]['num_ones']['pre-screening'])
                                total_zeros_screened_both += (self.node_z_summary[index]['num_zeros']['post-screening']- self.node_z_summary[index]['num_zeros']['pre-screening'])
            summary[level] = {'level': level, \
                              'num_nodes': len(level_indexes)}
            if self.perform_l0_scrn:
                if level == 0:
                    summary[level]['ones_screened'] = total_ones_screened
                    summary[level]['zeros_screened'] = total_zeros_screened
                    if self.BnBParams.branching == 'strong_act_with_scrn' or self.BnBParams.branching == \
                        'full_strong_act_with_scrn':
                        summary[level]['ones_screened_both'] = total_ones_screened_both
                        summary[level]['zeros_screened_both'] = total_zeros_screened_both
                else:
                    summary[level]['ones_screened'] = summary[level-1]['ones_screened']\
                                                             +total_ones_screened/2**level
                    summary[level]['zeros_screened'] = summary[level-1]['zeros_screened']\
                                                             +total_zeros_screened/2**level
                    if self.BnBParams.branching == 'strong_act_with_scrn' or self.BnBParams.branching == \
                        'full_strong_act_with_scrn':
                        summary[level]['ones_screened_both'] = summary[level-1]['ones_screened_both']\
                                                             +total_ones_screened_both/2**level
                        summary[level]['zeros_screened_both'] = summary[level-1]['zeros_screened_both']\
                                                             +total_zeros_screened_both/2**level
            self.level_summary[level]['ones_scrn_ratio'] = summary[level]['ones_screened']/self.p
            self.level_summary[level]['zeros_scrn_ratio'] = summary[level]['zeros_screened']/self.p
            if self.BnBParams.branching == 'strong_act_with_scrn' or self.BnBParams.branching == \
            'full_strong_act_with_scrn':
                self.level_summary[level]['ones_scrn_ratio_both'] = summary[level]['ones_screened_both']/self.p
                self.level_summary[level]['zeros_scrn_ratio_both'] = summary[level]['zeros_screened_both']/self.p
        return summary