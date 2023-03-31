##########
# L2_CD
##########

# def L2_CD_solve(X, y, l2, M, support, S_diag=None, warm_start=None, rel_tol=1e-4,cd_max_itr=100,kkt_max_itr=100,verbose=False):
#     if S_diag is None:
#         S_diag = np.linalg.norm(X,axis=0)**2
#     if warm_start is not None:
#         beta = np.copy(warm_start['beta'])
#         Xb = warm_start.get("Xb", X@beta)
#     else:
#         beta, Xb = trivial_soln(X)
    
#     active_set = support_to_active_set(support)
#     cost = get_L2_primal_cost(X, y, beta, Xb, l2, M, active_set)
#     cd_tol = rel_tol/2
#     if verbose:
#         print("cost", cost)
#     curiter = 0
#     while curiter < kkt_max_itr:
#         beta, cost, Xb = L2_CD(X, y, beta, cost, l2, M, S_diag, active_set, Xb, cd_tol, cd_max_itr, verbose)
#         if verbose:
#             print("iter", curiter+1)
#             print("cost", cost)
            
#         dual_cost = get_L2_dual_cost(X, y, beta, Xb, l2, M, active_set)
#         if verbose:
#             print("dual", dual_cost)
        
#         if (compute_relative_gap(cost, dual_cost) < rel_tol) or (cd_tol < 1e-8):
#             break
#         else:
#             cd_tol /= 100
#         curiter += 1
#     return beta, cost, Xb


##########
# L2_ASCD
##########
