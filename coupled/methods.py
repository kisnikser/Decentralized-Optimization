import numpy as np
from models import Model
from typing import Dict
import utils

def intermediate(num_steps: int, 
                 model: Model, 
                 params: Dict[str, float] = None):
    """
    Intermediate algorithm from the paper 
    "An Optimal Algorithm for Strongly Convex Minimization under Affine Constraints", 2022.
    It is a variant of the PAPC algorithm with Nesterov acceleration.
    
    Args:
        num_steps: int - Number of optimizer steps.
        model: Model - Model with oracle, which gives F, grad_F, etc.
        params: Dict[str, float] = None - Algorithm parameters.
    Returns:
        x_f: float - Solution.
        x_err: float - Distance to the actual solution.
        F_err: float - Function error.
        cons_err: float - Constraints error.
    """

    # set variables to the paper notation
    # "An Optimal Algorithm for Strongly Convex Minimization under Affine Constraints", 2022
    K = np.hstack((model.bA_prime, model.bW))
    W = K.T @ K
    b = model.bb_prime
    
    # set algorithm parameters
    lambda1 = utils.lambda_max(W) # can be chosen as lambda1 >= lambda_max
    lambda2 = utils.lambda_min_plus(W) # can be chosen as 0 < lambda2 <= lambda_min_plus
    chi = lambda1 / lambda2 # condition number of the W = K.T @ K
    params = {} if params is None else params
    tau = params.get('tau', min(1 - 1e-6, 1/2 * np.sqrt(chi/model.kappa)))
    assert tau > 0, "The parameter tau must be greater than 0"
    assert tau < 1, "The parameter tau must be less than 1"
    eta = params.get('eta', 1 / (4*tau*model.L))
    theta = params.get('theta', 1 / (eta*lambda1))
    alpha = params.get('alpha', model.mu)
    assert alpha > 0, "The parameter alpha must be greater than 0"

    # set the initial point
    #x_f = x = np.zeros(model.dim)
    xz_f = xz = np.zeros(model.dim + model.n * model.m) # for augmented function
    y = np.zeros(model.n * model.m)
    
    # get CVXPY solution
    xz_star, F_star = model.augmented_solution
    x_star = xz_star[:model.dim]
    
    # logging
    x_err = np.zeros(num_steps) # distance
    F_err = np.zeros(num_steps) # function error
    cons_err = np.zeros(num_steps) # constraints error
    
    for i in range(num_steps):
        xz_prev = xz # previous point
        xz_g = tau * xz + (1 - tau) * xz_f # point for gradient
        #g = model.grad_F(x_g) # calculate gradient
        g = model.grad_tildeF(xz_g) # calculate gradient of the augmented function
        xz_half = 1 / (1 + eta * alpha) * (xz - eta * (g - alpha * xz_g + K.T @ y)) # half point
        y = y + theta * (K @ xz_half - b)
        xz = 1 / (1 + eta * alpha) * (xz - eta * (g - alpha * xz_g + K.T @ y)) # next point
        xz_f = xz_g + 2 * tau / (2 - tau) * (xz - xz_prev) # point for function
        
        # add values to the logs
        x_f = xz_f[:model.dim]
        x_err[i] = np.linalg.norm(x_f - x_star) # ||x_f - x*||_2
        F_err[i] = model.tildeF(xz_f) - F_star # F(x_f) - F*
        cons_err[i] = np.linalg.norm(K @ xz_f - b) # ||K @ xz_f - b||_2
        
    return x_f, x_err, F_err, cons_err  


###################################################################################################


def chebyshev(z, K, b, N, lambda1, lambda2):
    """
    Chebyshev iteration.
    
    Args:
        z: np.ndarray - Initial point.
        K: np.ndarray - Matrix K (see notation in paper).
        b: np.ndarray - Vector b (see notation in paper).
        N: int - Number of steps.
        lambda1: float - first parameter (m.b. max eigenvalue).
        lambda2: float - second parameter (m.b. min positive eigenvalue).
    Returns:
        z: np.ndarray - Point after N steps.
    """
    assert lambda1 > 0, "lambda1 must be greater than 0"
    assert lambda2 > 0, "lambda2 must be greater than 0"
    
    rho = (lambda1 - lambda2)**2 / 16
    nu = (lambda1 + lambda2) / 2
    
    gamma = - nu / 2
    p = - K.T @ (K @ z - b) / nu
    z = z + p
        
    for _ in range(1, N):
        beta = rho / gamma
        gamma = - (nu + beta)
        p = (K.T @ (K @ z - b) + beta * p) / gamma
        z = z + p
            
    return z


###################################################################################################


def salim(num_steps: int, 
          model: Model, 
          params: Dict[str, float] = None):
    """
    Proposed algorithm 1 from the paper 
    "An Optimal Algorithm for Strongly Convex Minimization under Affine Constraints", 2022.
    
    Args:
        num_steps: int - Number of optimizer steps.
        model: Model - Model with oracle, which gives F, grad_F, etc.
        params: Dict[str, float] = None - Algorithm parameters.
    Returns:
        x_f: float - Solution.
        x_err: float - Distance to the actual solution.
        F_err: float - Function error.
        cons_err: float - Constraints error.
    """

    # set variables to the paper notation
    # "An Optimal Algorithm for Strongly Convex Minimization under Affine Constraints", 2022
    K = np.hstack((model.bA_prime, model.bW))
    W = K.T @ K
    b = model.bb_prime
    
    # set algorithm parameters
    lambda1 = utils.lambda_max(W) # can be chosen as lambda1 >= lambda_max
    lambda2 = utils.lambda_min_plus(W) # can be chosen as 0 < lambda2 <= lambda_min_plus
    chi = lambda1 / lambda2 # condition number of the W = K.T @ K
    params = {} if params is None else params
    tau = params.get('tau', min(1 - 1e-6, 1/2 * np.sqrt(19/(15*model.kappa))))
    assert tau > 0, "The parameter tau must be greater than 0"
    assert tau < 1, "The parameter tau must be less than 1"
    eta = params.get('eta', 1 / (4*tau*model.L))
    theta = params.get('theta', 15 / (19*eta))
    alpha = params.get('alpha', model.mu)
    assert alpha > 0, "The parameter alpha must be greater than 0"
    N = int(np.sqrt(chi)) + 1 # can be chosen as N >= sqrt(chi)

    # set the initial point
    #x_f = x = np.zeros(model.dim)
    xz_f = xz = np.zeros(model.dim + model.n * model.m) # for augmented function
    u = np.zeros(model.dim + model.n * model.m)
    
    # get CVXPY solution
    xz_star, F_star = model.augmented_solution
    x_star = xz_star[:model.dim]
    
    # logging
    x_err = np.zeros(num_steps) # distance
    F_err = np.zeros(num_steps) # function error
    cons_err = np.zeros(num_steps) # constraints error
    
    for i in range(num_steps):
        xz_prev = xz # previous point
        xz_g = tau * xz + (1 - tau) * xz_f # point for gradient
        #g = model.grad_F(x_g) # calculate gradient
        g = model.grad_tildeF(xz_g) # calculate gradient of the augmented function
        xz_half = 1 / (1 + eta * alpha) * (xz - eta * (g - alpha * xz_g + u)) # half point
        r = theta * (xz_half - chebyshev(xz_half, K, b, N, lambda1, lambda2))
        u = u + r
        xz = xz_half - eta / (1 + eta * alpha) * r # next point
        xz_f = xz_g + 2 * tau / (2 - tau) * (xz - xz_prev) # point for function
        
        # add values to the logs
        x_f = xz_f[:model.dim]
        x_err[i] = np.linalg.norm(x_f - x_star) # ||x_f - x*||_2
        F_err[i] = model.tildeF(xz_f) - F_star # F(x_f) - F*
        cons_err[i] = np.linalg.norm(K @ xz_f - b) # ||Kx_f - b||_2
        
    return x_f, x_err, F_err, cons_err  