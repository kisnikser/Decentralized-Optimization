import numpy as np
import cvxpy as cp
import scipy as sp
from models import Model, ExampleModel
from typing import Dict
import utils

def intermediate(num_steps: int, 
                 model: ExampleModel, 
                 params: Dict[str, float] = None):
    """
    Intermediate algorithm from the paper 
    "An Optimal Algorithm for Strongly Convex Minimization under Affine Constraints", 2022.
    It is a variant of the PAPC algorithm with Nesterov acceleration.
    
    Args:
        num_steps: int - Number of optimizer steps.
        model: ExampleModel - Model with oracle, which gives F, grad_F, etc.
        params: Dict[str, float] = None - Algorithm parameters.
    Returns:
        x_f: float - Solution.
        x_err: np.ndarray - Sequence of distances to the actual solution.
        F_err: np.ndarray - Sequence of function error.
        cons_err: np.ndarray - Sequence of constraints error.
        primal_dual_err: np.ndarray - Sequence of primal-dual optimality condition error.
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
    tau = params.get('tau', min(1, 1/2 * np.sqrt(chi/model.kappa)))
    assert tau >= 0, "The parameter tau must be greater than 0"
    assert tau <= 1, "The parameter tau must be less than 1"
    eta = params.get('eta', 1 / (4*tau*model.L))
    theta = params.get('theta', 1 / (eta*lambda1))
    alpha = params.get('alpha', model.mu)
    assert alpha > 0, "The parameter alpha must be greater than 0"

    # set the initial point
    #x_f = x = np.zeros(model.dim)
    xz_f = xz = np.zeros(model.dim + model.n * model.m) # for augmented function
    y = np.zeros(model.n * model.m)
    
    # get CVXPY solution
    xz_star, F_star = model.solution
    x_star = xz_star[:model.dim]
    
    # logging
    x_err = np.zeros(num_steps) # distance
    F_err = np.zeros(num_steps) # function error
    cons_err = np.zeros(num_steps) # constraints error
    primal_dual_err = np.zeros(num_steps) # primal-dual optimality condition error
    
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
        x_err[i] = np.linalg.norm(x_f - x_star)**2 # ||x_f - x*||_2^2
        F_err[i] = abs(model.tildeF(xz_f) - F_star) # |\tilde{F}(xz_f) - \tilde{F}*|
        cons_err[i] = np.linalg.norm(K @ xz_f - b) # ||K @ xz_f - b||_2
        primal_dual_err[i] = np.linalg.norm(K.T @ y + model.grad_tildeF(xz_f))
        
    return x_f, x_err, F_err, cons_err, primal_dual_err


###################################################################################################


def chebyshev(z_0, K, b, N, lambda1, lambda2):
    """
    Chebyshev iteration.
    
    Args:
        z_0: np.ndarray - Initial point.
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
    p = - K.T @ (K @ z_0 - b) / nu
    z = z_0 + p
        
    for _ in range(1, N):
        beta = rho / gamma
        gamma = - (nu + beta)
        p = (K.T @ (K @ z - b) + beta * p) / gamma
        z = z + p
            
    return z

#--------------------------------------------------------------------------------------------------

def salim(num_steps: int, 
          model: ExampleModel, 
          params: Dict[str, float] = None):
    """
    Proposed algorithm 1 from the paper 
    "An Optimal Algorithm for Strongly Convex Minimization under Affine Constraints", 2022.
    
    Args:
        num_steps: int - Number of optimizer steps.
        model: ExampleModel - Model with oracle, which gives F, grad_F, etc.
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
    tau = params.get('tau', min(1, 1/2 * np.sqrt(19/(15*model.kappa))))
    assert tau >= 0, "The parameter tau must be greater than 0"
    assert tau <= 1, "The parameter tau must be less than 1"
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
    xz_star, F_star = model.solution
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
        x_err[i] = np.linalg.norm(x_f - x_star)**2 # ||x_f - x*||_2^2
        F_err[i] = abs(model.tildeF(xz_f) - F_star) # |\tilde{F}(x_f) - \tilde{F}*|
        cons_err[i] = np.linalg.norm(K @ xz_f - b) # ||K @ xz_f - b||_2
        
    return x_f, x_err, F_err, cons_err  


###################################################################################################


def get_argmin_DPMM(x_k: np.ndarray,
                    y: np.ndarray,
                    alpha: np.ndarray,
                    gamma: np.ndarray,
                    model: Model,
                    mode: str = 'newton'):
    """
    Solve argmin subproblem in DPMM for our Example problem.
    As we have quadratic problem, we do just one step on Newton method.
    Also we can solve it using CVXPY.
    
    Args:
        x_k: np.ndarray - Value of primal variable vector x from previous step.
        y: np.ndarray - Value of vector y from previous step.
        alpha: np.ndarray - Vector of parameters alpha.
        gamma: np.ndarray - Vector of parameters gamma.
        model: Model - Model with oracle, which gives F, grad_F, etc.
        mode: str = 'newton' - Use newton or CVXPY.
    Returns:
        sol: np.ndarray - Solution for argmin subproblem.
    """
    x_k_array = model._split_vector(x_k)
    y_array = np.split(y, np.cumsum([model.m for _ in range(model.n)])[:-1])
    sol = []

    if mode == 'newton':
        for i in range(model.n):
            # calculate gradient function
            grad = lambda x: (
                model.C[i].T @ model.C[i] @ x - model.C[i].T @ model.d[i] + model.theta * x
                + model.A[i].T @ (y_array[i] + gamma[i] * (model.A[i] @ x - model.b[i]))
                + 1 / alpha[i] * (x - x_k_array[i])
            )
            
            # calculate hessian matrix
            hess = lambda x: (
                model.C[i].T @ model.C[i] + model.theta * np.identity(model.dims[i])
                + gamma[i] * model.A[i].T @ model.A[i]
                + 1 / alpha[i] * np.identity(model.dims[i])
            ) 
            
            # get a solution by one newton step
            sol.extend(x_k_array[i] - np.linalg.inv(hess(x_k_array[i])) @ grad(x_k_array[i]))
            
        sol = np.array(sol)
        
    else:
        raise NotImplementedError
    
    return sol


#--------------------------------------------------------------------------------------------------


def DPMM(num_steps: int, 
         model: ExampleModel, 
         params: Dict[str, float] = None):
    """
    Decentralized Proximal Method of Multipliers (DPMM) from the paper
    "Decentralized Proximal Method of Multipliers for Convex Optimization with Coupled Constraints", 2023.
    
    Args:
        num_steps: int - Number of optimizer steps.
        model: ExampleModel - Model with oracle, which gives F, grad_F, etc.
        params: Dict[str, float] = None - Algorithm parameters.
    Returns:
        x_f: float - Solution.
        x_err: float - Distance to the actual solution.
        F_err: float - Function error.
        cons_err: float - Constraints error.
    """
    # set variables to the paper notation
    I_n = np.identity(model.dim)
    bL = model.bW
    G_d = lambda x: model.bA_prime @ x - model.bb_prime
    
    # set algorithm parameters
    params = {} if params is None else params
    
    theta = params.get('theta', np.ones(model.n))
    assert np.all(theta > 0) and np.all(theta < 2), "Parameter theta must be greater than 0 and less than 2"
    Theta = sp.linalg.block_diag(*[theta[i] * np.identity(model.dims[i]) for i in range(model.n)])
    
    alpha = params.get('alpha', np.ones(model.n))
    assert np.all(alpha > 0), "Parameter alpha must be greater than 0"
    #Upsilon = sp.linalg.block_diag(*[alpha[i] * np.identity(model.dims[i]) for i in range(model.n)])
    
    gamma = params.get('gamma', np.ones(model.n))
    assert np.all(gamma > 0), "Parameter gamma must be greater than 0"
    Gamma = sp.linalg.block_diag(*[gamma[i] * np.identity(model.m) for i in range(model.n)])
    
    beta = params.get('beta', min(1 / (gamma * utils.lambda_max(bL))) / 2)
    assert beta > 0 and beta < min(1 / (gamma * utils.lambda_max(bL))), "Wrong parameter beta"
    
    # set the initial point
    x = np.zeros(model.dim)
    y = np.zeros(model.n * model.m)
    Lambda = np.zeros(model.n * model.m)
    
    # get CVXPY solution
    x_star, F_star = model.solution_initial
    
    # logging
    x_err = np.zeros(num_steps) # distance
    F_err = np.zeros(num_steps) # function error
    cons_err = np.zeros(num_steps) # constraints error
    
    for i in range(num_steps):
        # algorithm step
        x_hat = get_argmin_DPMM(x, y - Gamma @ Lambda, alpha, gamma, model, mode='newton')
        y_hat = y - Gamma @ Lambda + Gamma @ G_d(x_hat)
        x = (I_n - Theta) @ x + Theta @ x_hat
        Lambda_prev = Lambda
        Lambda = Lambda_prev + beta * bL @ y_hat
        y = y_hat + Gamma @ (Lambda_prev - Lambda)
        
        # add values to the logs
        x_err[i] = np.linalg.norm(x - x_star)**2 # ||x - x*||_2^2
        F_err[i] = abs(model.F(x) - F_star) # |F(x) - F*|
        cons_err[i] = np.linalg.norm(model.bA @ x - model.bb) # ||bA @ x - bb||_2
        
    return x, x_err, F_err, cons_err


###################################################################################################


def get_argmin_TrackingADMM(x_k: np.ndarray,
                            d_k: np.ndarray,
                            lmbd_k: np.ndarray,
                            c: float,
                            model: Model,
                            mode: str = 'newton'):
    """
    Solve argmin subproblem in Tracking-ADMM for our Example problem.
    As we have quadratic problem, we do just one step on Newton method.
    Also we can solve it using CVXPY.
    
    Args:
        x_k: np.ndarray - Value of primal variable vector x from previous step.
        d_k: np.ndarray - Value of vector d from previous step.
        lmbd_k: np.ndarray - Value of vector lmbd from previous step.
        c: float - Parameter for augmentation.
        model: Model - Model with oracle, which gives F, grad_F, etc.
        mode: str = 'newton' - Use newton or CVXPY.
    Returns:
        sol: np.ndarray - Solution for argmin subproblem.
    """
    # set variables to the paper notation
    W = np.kron(model.mixing_matrix, np.identity(model.m))
    A_d = model.bA_prime
    
    if mode == 'newton':
        # calculate gradient function
        grad = lambda x: (
            model.grad_F(x)
            + A_d.T @ W @ lmbd_k
            + c * A_d.T @ (A_d @ (x - x_k) + W @ d_k)
        )
        
        # calculate hessian matrix
        hess = lambda x: model.hess_F(x) + c * A_d.T @ A_d

        # get a solution by one newton step
        sol = x_k - np.linalg.inv(hess(x_k)) @ grad(x_k)
        
    elif mode == 'cvxpy':
        x = cp.Variable(model.dim)
        objective = cp.Minimize(
            1/2 * cp.sum_squares(model.bC @ x - model.bd) 
            + model.theta/2 * cp.sum_squares(x)
            + (W @ lmbd_k).T @ A_d @ x
            + c/2 * cp.sum_squares(A_d @ (x - x_k) + W @ d_k)
        )
        prob = cp.Problem(objective)
        prob.solve()
        sol = x.value
        
    else:
        raise NotImplementedError
    
    return sol


#--------------------------------------------------------------------------------------------------


def TrackingADMM(num_steps: int, 
                 model: ExampleModel, 
                 params: Dict[str, float] = None):
    """
    Tracking-ADMM algorithm from the paper
    "Tracking-ADMM for distributed constraint-coupled optimization", 2020.
    
    Args:
        num_steps: int - Number of optimizer steps.
        model: ExampleModel - Model with oracle, which gives F, grad_F, etc.
        params: Dict[str, float] = None - Algorithm parameters.
    Returns:
        x_f: float - Solution.
        x_err: float - Distance to the actual solution.
        F_err: float - Function error.
        cons_err: float - Constraints error.
    """
    # set variables to the paper notation
    W = np.kron(model.mixing_matrix, np.identity(model.m))
    A_d = model.bA_prime
    
    # set algorithm parameters
    params = {} if params is None else params
    c = params.get('c', 1e-3)
    assert c > 0, "Parameter c must be greater than 0"
    
    # set the initial point
    x = np.zeros(model.dim)
    d = np.zeros(model.n * model.m)
    lmbd = np.zeros(model.n * model.m)
    
    # get CVXPY solution
    x_star, F_star = model.solution_initial
    
    # logging
    x_err = np.zeros(num_steps) # distance
    F_err = np.zeros(num_steps) # function error
    cons_err = np.zeros(num_steps) # constraints error
    
    for i in range(num_steps):
        # algorithm step
        x_prev = x
        x = get_argmin_TrackingADMM(x_prev, d, lmbd, c, model, mode='newton')
        d = W @ d + A_d @ (x - x_prev)
        lmbd = W @ lmbd + c * d
        
        # add values to the logs
        x_err[i] = np.linalg.norm(x - x_star)**2 # ||x - x*||_2^2
        F_err[i] = abs(model.F(x) - F_star) # |F(x) - F*|
        cons_err[i] = np.linalg.norm(model.bA @ x - model.bb) # ||bA @ x - bb||_2
        
    return x, x_err, F_err, cons_err  