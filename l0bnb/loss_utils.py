import numpy as np
from numba import njit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

@njit(cache=True)
def compute_loss(loss_name, y, X, beta, Xb, delta=1.):
    if loss_name == 'lstsq':
        tmp = (y-Xb)
        loss = 0.5*(tmp@tmp)
    elif loss_name == 'logistic':
        loss = np.sum(np.log1p(np.exp(-y*Xb)))
    elif loss_name == 'sqhinge':
        loss = np.sum(np.maximum(1-y*Xb,0)**2)
    elif loss_name == 'hbhinge':
        tmp = 1-y*Xb
        loss = np.sum(np.where(tmp>delta, tmp-delta/2, np.where(tmp<=0,0,tmp**2/2/delta)))
    else:
        loss = 0.
    return loss

@njit(cache=True)
def compute_grad(loss_name, y, X, beta, Xb, delta=1.):
    if loss_name == 'lstsq':
        grad = (Xb-y)@X
    elif loss_name == 'logistic':
        grad = -(y/(np.exp(y*Xb)+1))@X
    elif loss_name == 'sqhinge':
        grad = -2*(y*np.maximum(1-y*Xb,0))@X
    elif loss_name == 'hbhinge':
        grad = -(y*np.maximum(np.minimum((1-y*Xb)/delta,1),0))@X
    else:
        grad = np.zeros_like(beta)
    return grad


@njit(cache=True)
def compute_grad_Lipschitz_sigma(loss_name, delta=1.):
    if loss_name == 'lstsq':
        sigma = 1.
    elif loss_name == 'logistic':
        sigma = .25
    elif loss_name == 'sqhinge':
        sigma = 2.
    elif loss_name == 'hbhinge':
        sigma = 1/delta
    else:
        sigma = 0.
    return sigma

@njit(cache=True)
def compute_coordinate_grad_Lipschitz(loss_name, i, y, X, beta, Xb, S_diag, delta=1.):
    if loss_name == 'lstsq':
        grad_i = (Xb-y)@X[:,i]
        Li = S_diag[i]
    elif loss_name == 'logistic':
        grad_i = -(y/(np.exp(y*Xb)+1))@X[:,i]
        Li = S_diag[i]/4
    elif loss_name == 'sqhinge':
        grad_i = -2*(y*np.maximum(1-y*Xb,0))@X[:,i]
        Li = 2*S_diag[i]
    elif loss_name == 'hbhinge':
        grad_i = -(y*np.maximum(np.minimum((1-y*Xb)/delta,1),0))@X[:,i]
        Li = S_diag[i]/delta
    else:
        grad_i = 0.
        Li = S_diag[i]
    return grad_i, Li
    
    
        
@njit(cache=True)
def compute_loss_grad(loss_name, y, X, beta, Xb, delta=1.):
    if loss_name == 'lstsq':
        tmp = (y-Xb)
        loss = 0.5*(tmp@tmp)
        grad = -tmp@X
    elif loss_name == 'logistic':
        tmp = y*Xb
        loss = np.sum(np.log1p(np.exp(-tmp)))
        grad = -(y/(np.exp(tmp)+1))@X
    elif loss_name == 'sqhinge':
        tmp = np.maximum(1-y*Xb,0)
        loss = np.sum(tmp**2)
        grad = -2*(y*tmp)@X
    elif loss_name == 'hbhinge':
        tmp = 1-y*Xb
        loss = np.sum(np.where(tmp>delta, tmp-delta/2, np.where(tmp<=0,0,tmp**2/2/delta)))
        grad = -(y*np.maximum(np.minimum(tmp/delta,1),0))@X
    else:
        loss = 0.
        grad = np.zeros_like(beta)
    return loss, grad
    
@njit(cache=True)
def compute_grad_theta(loss_name, y, X, beta, Xb, delta=1.):
    if loss_name == 'lstsq':
        theta = y-Xb
    elif loss_name == 'logistic':
        theta = y/(np.exp(y*Xb)+1)
    elif loss_name == 'sqhinge':
        theta =  2*(y*np.maximum(1-y*Xb,0))
    elif loss_name == 'hbhinge':
        theta =  y*np.maximum(np.minimum((1-y*Xb)/delta,1),0)
    else:
        theta =  np.zeros_like(y)
    return -theta@X, theta
    