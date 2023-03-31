import numpy as np
from scipy.stats import norm as Normal
    
def generate_synthetic(n,p,k,model='regression',rho=0,snr=10,supp_loc='random',nonneg=True,beta_min=1.,scale=1.,rng=None):
    """Generate a synthetic regression dataset.

    The data matrix X is sampled from a multivariate gaussian and
    the error term epsilon is sampled independent from a normal
    distribution. Depending on the model, the response y is generated based on X@beta + epsilon, 
    where beta is a sparse vector, where the abs values of all the nonzero elements are between beta_min and 1.

    Inputs:
        n: Number of samples.
        p: Number of features.
        k: Number of nonzeros in beta
        rho: float, default 0
            Correlation parameter.
        snr: float, default 10
            Signal-to-noise ratio.
        supp_loc: str, default 'random'
            The way to generate support locations: 'random' or 'equispaced'
        nonneg: bool, default True
            whether the nonzero elements of beta are nonnegative.
        beta_min: float between 0 and 1, default 1
            lower bound of abs values of nonzeros of beta.
        rng: None, int, or random generator
            If rng is None, then it becomes np.random
            If rng is int, then it becomes np.random.RandomState(rng)
        model: str
            'regression': y = X@beta + epsilon
            'logistic': Pr(y=1) = 1/(1+exp(- X@beta - epsilon))
            'probit': Pr(y=1) = Phi(X@beta + epsilon), where Phi is CDF of standard normal
            'SVM': y = 1 iff X@beta + epsilon > 0
    Returns:
        X: The data matrix.
        y: The response vector.
        beta: The true vector of coefficients.
    """
    assert model in {'regression', 'logistic', 'probit', 'SVM'}
    assert snr > 0
    assert beta_min <= 1. and beta_min >= 0.
    if rng is None:
        rng = np.random
    elif type(rng) == int:
        rng = np.random.RandomState(rng)
    
    X = rng.randn(n, p)
    if rho != 0:
        X = X*np.sqrt(1-rho) + np.sqrt(rho)*rng.randn(n,1)
    beta = np.zeros(p)
    if beta_min >= 1:
        beta_supp = 1
    else:
        beta_supp = rng.rand(k)*(1-beta_min)+beta_min
    if supp_loc == 'random':
        support = rng.choice(p,k,replace=False)
    elif supp_loc == 'equispaced':
        support = np.array([int(i * (p / k)) for i in range(k)],dtype=int)
    if nonneg:
        beta[support] = beta_supp
    else:
        beta[support] = (2*rng.choice(2,k)-1)*beta_supp
    
    mu = X@beta
    nse_std = np.std(mu)/np.sqrt(snr)
    epsilon = rng.randn(n)*nse_std
    y = (mu+epsilon)*scale
    
    if model == 'regression':
        pass
    elif model == 'logistic':
        y = (rng.rand(n) <= 1/(1+np.exp(-y)))
        y = 2*y - 1
    elif model == 'probit':
        y = (rng.rand(n) <= Normal.cdf(y))
        y = 2*y - 1
    elif model == 'SVM':
        y = np.where(y>0, 1, -1)
    y = y.astype(float)
    return X, y, beta #, support

def generate_synthetic_new(n,p,k,model='regression',snr=10,supp_loc='equispaced',nonneg=True,scale=1.,rng=None):
    """Generate a synthetic regression dataset.

    The data matrix X is sampled from a multivariate gaussian and
    the error term epsilon is sampled independent from a normal
    distribution. Depending on the model, the response y is generated based on X@beta + epsilon, 
    where beta is a sparse vector, where the abs values of all the nonzero elements are between beta_min and 1.

    Inputs:
        n: Number of samples.
        p: Number of features.
        k: Number of nonzeros in beta
        snr: float, default 10
            Signal-to-noise ratio.
        supp_loc: str, default 'random'
            The way to generate support locations: 'random' or 'equispaced'
        nonneg: bool, default True
            whether the nonzero elements of beta are nonnegative.
        rng: None, int, or random generator
            If rng is None, then it becomes np.random
            If rng is int, then it becomes np.random.RandomState(rng)
        model: str
            'regression': y = X@beta + epsilon
    Returns:
        X: The data matrix.
        y: The response vector.
        beta: The true vector of coefficients.
        l0
        M
    """
    assert model in {'regression'}
    assert snr > 0
    if rng is None:
        rng = np.random
    elif type(rng) == int:
        rng = np.random.RandomState(rng)
    
    X = rng.randn(n, p)
    X = X/np.linalg.norm(X,axis=0)
    
    beta = np.zeros(p)
    beta_supp = rng.choice((-1,1),k)*(1+np.abs(rng.randn(k)))
    if supp_loc == 'random':
        support = rng.choice(p,k,replace=False)
    elif supp_loc == 'equispaced':
        support = np.array([int(i * (p / k)) for i in range(k)],dtype=int)
    beta[support] = beta_supp
    
    Xb = X@beta
    nse_std = np.linalg.norm(Xb)/np.sqrt(snr*n)
    epsilon = rng.randn(n)*nse_std
    y = Xb+epsilon
    y = y.astype(float)
    
    l0 = 2*nse_std*np.log(p/k-1)
    M = 1.5*np.max(np.abs(y@X))
    
    return X, y, beta, l0, M

def preprocess(X,y):
    n,p = X.shape
    S_diag = np.linalg.norm(X, axis=0)**2
    return n, p, X, y, S_diag