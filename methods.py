import numpy as np
import cvxpy as cp
import scipy as sp
from models import Model, ExampleModel, VFL
from typing import Dict, Optional
import utils
import time


###################################################################################################


def TrackingADMM(num_steps: int, 
                 model: Model, 
                 params: Optional[Dict[str, float]] = None):
    """
    Tracking-ADMM algorithm from the paper
    "Tracking-ADMM for distributed constraint-coupled optimization", 2020.
    
    Args:
        num_steps: int - Number of optimizer steps.
        model: Model - Model with oracle, which gives F, grad_F, etc.
        params: Optional[Dict[str, float]] = None - Algorithm parameters.
    Returns:
        x_f: float - Solution.
        x_err: float - Distance to the actual solution.
        F_err: float - Function error.
        cons_err: float - Constraints error.
        ts: np.ndarray - Sequence of time taken for previous iterations.
    """
    
    if isinstance(model, ExampleModel):
        get_argmin_TrackingADMM = get_argmin_TrackingADMM_example
    elif isinstance(model, VFL):
        get_argmin_TrackingADMM = get_argmin_TrackingADMM_vfl
    else:
        raise NotImplementedError
    
    # set variables to the paper notation
    W = np.kron(model.mixing_matrix, np.identity(model.m))
    A_d = model.bA
    
    # set algorithm parameters
    params = {} if params is None else params
    c = params.get('c', 1e-6)
    assert c > 0, "Parameter c must be greater than 0"
    
    # set the initial point
    x = np.zeros(model.dim)
    xs = [np.zeros(model.dimensions[i]) for i in range(model.n)]
    d = np.hstack([Ai @ xi0 - bi for (Ai, xi0, bi) in zip(model.A, xs, model.b)])
    lmbd = np.zeros(model.n * model.m)
    
    # get CVXPY solution
    x_star, F_star = model.solution
    
    # logging
    x_err = np.zeros(num_steps) # distance
    F_err = np.zeros(num_steps) # function error
    cons_err = np.zeros(num_steps) # constraints error
    
    ts = []
    start = time.time()
    
    for i in range(num_steps):
        # algorithm step
        x_prev = x
        x = get_argmin_TrackingADMM(x_prev, d, lmbd, c, model)
        d = W @ d + A_d @ (x - x_prev)
        lmbd = W @ lmbd + c * d
        
        end = time.time()
        ts.append(end - start)
        
        # add values to the logs
        x_err[i] = np.linalg.norm(x - x_star)**2 # ||x - x*||_2^2
        F_err[i] = abs(model.F(x) - F_star) # |F(x) - F*|
        cons_err[i] = np.linalg.norm(model.A_hstacked @ x - model.b_sum) # ||bA @ x - bb||_2
        
    return x, x_err, F_err, cons_err, ts

#--------------------------------------------------------------------------------------------------

# Example model
def get_argmin_TrackingADMM_example(x_k: np.ndarray,
                                    d_k: np.ndarray,
                                    lmbd_k: np.ndarray,
                                    c: float,
                                    model: ExampleModel):
    """
    Solve argmin subproblem in Tracking-ADMM for our Example problem.
    As we have quadratic problem, we do just one step on Newton method.
    Also we can solve it using CVXPY.
    
    Args:
        x_k: np.ndarray - Value of primal variable vector x from previous step.
        d_k: np.ndarray - Value of vector d from previous step.
        lmbd_k: np.ndarray - Value of vector lmbd from previous step.
        c: float - Parameter for augmentation.
        model: VFL - Model with oracle, which gives F, grad_F, etc.
    Returns:
        sol: np.ndarray - Solution for argmin subproblem.
    """
    # set variables to the paper notation
    W = np.kron(model.mixing_matrix, np.identity(model.m))
    A_d = model.bA
        
    A = model.bCT_bC + model.theta * np.identity(model.dim) + c * A_d.T @ A_d
    
    b = (
        model.bCT_bd
        - A_d.T @ W @ lmbd_k
        + c * A_d.T @ (A_d @ x_k - W @ d_k)
    )
    
    x = np.linalg.solve(A, b)
        
    return x

#--------------------------------------------------------------------------------------------------

# VFL
def get_argmin_TrackingADMM_vfl(x_k: np.ndarray,
                                d_k: np.ndarray,
                                lmbd_k: np.ndarray,
                                c: float,
                                model: VFL,
                                mode: str = 'newton'):
    """
    Solve argmin subproblem in Tracking-ADMM for our VFL problem.
    As we have quadratic problem, we do just one step on Newton method.
    Also we can solve it using CVXPY.
    
    Args:
        x_k: np.ndarray - Value of primal variable vector x from previous step.
        d_k: np.ndarray - Value of vector d from previous step.
        lmbd_k: np.ndarray - Value of vector lmbd from previous step.
        c: float - Parameter for augmentation.
        model: VFL - Model with oracle, which gives F, grad_F, etc.
        mode: str = 'newton' - Use newton or CVXPY.
    Returns:
        sol: np.ndarray - Solution for argmin subproblem.
    """
    # set variables to the paper notation
    W = np.kron(model.mixing_matrix, np.identity(model.m))
    A_d = model.bA
    
    A = model.hess_F() + c * A_d.T @ A_d
    
    b = (
        model._rearrange_vector(np.hstack((model.l, np.zeros(model.n * model.d))))
        - A_d.T @ W @ lmbd_k
        + c * A_d.T @ (A_d @ x_k - W @ d_k)
    )
    
    x = np.linalg.solve(A, b)
    
    return x


###################################################################################################


def DPMM(num_steps: int, 
         model: Model, 
         params: Optional[Dict[str, float]] = None):
    """
    Decentralized Proximal Method of Multipliers (DPMM) from the paper
    "Decentralized Proximal Method of Multipliers for Convex Optimization with Coupled Constraints", 2023.
    
    Args:
        num_steps: int - Number of optimizer steps.
        model: Model - Model with oracle, which gives F, grad_F, etc.
        params: Optional[Dict[str, float]] = None - Algorithm parameters.
    Returns:
        x_f: float - Solution.
        x_err: float - Distance to the actual solution.
        F_err: float - Function error.
        cons_err: float - Constraints error.
        ts: np.ndarray - Sequence of time taken for previous iterations.
    """
    
    if isinstance(model, ExampleModel):
        get_argmin_DPMM = get_argmin_DPMM_example
    elif isinstance(model, VFL):
        get_argmin_DPMM = get_argmin_DPMM_vfl
    else:
        raise NotImplementedError
    
    # set variables to the paper notation
    I_n = np.identity(model.dim)
    bL = model.bW
    G_d = lambda x: model.bA @ x - model.bb
    
    # set algorithm parameters
    params = {} if params is None else params
    
    theta = params.get('theta', np.ones(model.n))
    assert np.all(theta > 0) and np.all(theta < 2), "Parameter theta must be greater than 0 and less than 2"
    Theta = sp.linalg.block_diag(*[theta[i] * np.identity(model.dimensions[i]) for i in range(model.n)])
    
    alpha = params.get('alpha', np.ones(model.n))
    assert np.all(alpha > 0), "Parameter alpha must be greater than 0"
    #Upsilon = sp.linalg.block_diag(*[alpha[i] * np.identity(model.d) for i in range(model.n)])
    
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
    x_star, F_star = model.solution
    
    # logging
    x_err = np.zeros(num_steps) # distance
    F_err = np.zeros(num_steps) # function error
    cons_err = np.zeros(num_steps) # constraints error
    
    ts = []
    start = time.time()
    
    for i in range(num_steps):
        # algorithm step
        x_hat = get_argmin_DPMM(x, y - Gamma @ Lambda, alpha, gamma, model)
        y_hat = y - Gamma @ Lambda + Gamma @ G_d(x_hat)
        x = (I_n - Theta) @ x + Theta @ x_hat
        Lambda_prev = Lambda
        Lambda = Lambda_prev + beta * bL @ y_hat
        y = y_hat + Gamma @ (Lambda_prev - Lambda)
        
        end = time.time()
        ts.append(end - start)
        
        # add values to the logs
        x_err[i] = np.linalg.norm(x - x_star)**2 # ||x - x*||_2^2
        F_err[i] = abs(model.F(x) - F_star) # |F(x) - F*|
        cons_err[i] = np.linalg.norm(model.A_hstacked @ x - model.b_sum) # ||bA @ x - bb||_2
        
    return x, x_err, F_err, cons_err, ts

#--------------------------------------------------------------------------------------------------

# Example model
def get_argmin_DPMM_example(x_k: np.ndarray,
                            y: np.ndarray,
                            alpha: np.ndarray,
                            gamma: np.ndarray,
                            model: ExampleModel):
    """
    Solve argmin subproblem in DPMM for our Example problem.
    As we have quadratic problem, we do just one step on Newton method.
    Also we can solve it using CVXPY.
    
    Args:
        x_k: np.ndarray - Value of primal variable vector x from previous step.
        y: np.ndarray - Value of vector y from previous step.
        alpha: np.ndarray - Vector of parameters alpha.
        gamma: np.ndarray - Vector of parameters gamma.
        model: ExampleModel - Model with oracle, which gives F, grad_F, etc.
    Returns:
        sol: np.ndarray - Solution for argmin subproblem.
    """
    x_k_array = np.split(x_k, np.cumsum([model.d for _ in range(model.n)])[:-1])
    y_array = np.split(y, np.cumsum([model.m for _ in range(model.n)])[:-1])
    
    x = []

    for i in range(model.n):
        
        A = (
            model.C[i].T @ model.C[i]
            + gamma[i] * model.A[i].T @ model.A[i]
            + (model.theta + 1 / alpha[i]) * np.identity(model.d)
        )
        
        b = (
            model.C[i].T @ model.d_[i]
            + model.A[i].T @ (gamma[i] * model.b[i] - y_array[i])
            + 1 / alpha[i] * x_k_array[i]
        )
        
        x.extend(np.linalg.solve(A, b))
        
    x = np.array(x)
    
    return x

#--------------------------------------------------------------------------------------------------

# VFL
def get_argmin_DPMM_vfl(x_k: np.ndarray,
                        y: np.ndarray,
                        alpha: np.ndarray,
                        gamma: np.ndarray,
                        model: VFL):
    """
    Solve argmin subproblem in DPMM for our VFL problem.
    As we have quadratic problem, we do just one step on Newton method.
    Also we can solve it using CVXPY.
    
    Args:
        x_k: np.ndarray - Value of primal variable vector x from previous step.
        y: np.ndarray - Value of vector y from previous step.
        alpha: np.ndarray - Vector of parameters alpha.
        gamma: np.ndarray - Vector of parameters gamma.
        model: ExampleModel - Model with oracle, which gives F, grad_F, etc.
    Returns:
        sol: np.ndarray - Solution for argmin subproblem.
    """
        
    cumsum = np.cumsum(model.dimensions)[:-1]
    x_k_array = np.split(x_k, cumsum)
    y_array = np.split(y, model.n)
    
    if model.labels_distribution:
        cumsum = np.cumsum(model.num_samples)[:-1]
        l_array = np.split(model.l, cumsum)
    else:
        l_array = [model.l]
        
    x = []
    
    cumsum = np.cumsum([0, *model.dimensions])
    
    for i in range(model.n):
        
        left = cumsum[i]
        right = cumsum[i+1]

        hess_f = model.hess_F()[left:right, left:right]
        
        if model.labels_distribution or i == 0:
            resid_f = np.hstack((np.zeros(model.d), l_array[i]))
        else:
            resid_f = np.zeros(model.d)
            
        A = (
            hess_f
            + gamma[i] * model.A[i].T @ model.A[i]
            + 1 / alpha[i] * np.identity(model.dimensions[i])
        )
        
        b = (
            resid_f
            + model.A[i].T @ (gamma[i] * model.b[i] - y_array[i])
            + 1 / alpha[i] * x_k_array[i]
        )
                
        x.extend(np.linalg.solve(A, b))
            
    x = np.array(x)
    
    return x


###################################################################################################


def APAPC(num_steps: int, 
                 model: Model, 
                 params: Dict[str, float] = None):
    """
    Intermediate algorithm from the paper 
    "An Optimal Algorithm for Strongly Convex Minimization under Affine Constraints", 2022.
    It is a variant of the PAPC algorithm with Nesterov acceleration (APAPC).
    
    Args:
        num_steps: int - Number of optimizer steps.
        model: Model - Model with oracle, which gives F, grad_F, etc.
        params: Dict[str, float] = None - Algorithm parameters.
    Returns:
        x: float - Solution.
        x_err: np.ndarray - Sequence of distances to the actual solution.
        F_err: np.ndarray - Sequence of function error.
        cons_err: np.ndarray - Sequence of constraints error.
        primal_dual_err: np.ndarray - Sequence of primal-dual optimality condition error.
        ts: np.ndarray - Sequence of time taken for previous iterations.
    """
    
    bB = np.hstack((model.bA, model.gamma * model.bW))
    mu_B = utils.get_s2min_plus(bB)
    L_B = utils.get_s2max(bB)
    kappa_B = L_B / mu_B
    
    # set algorithm parameters
    params = {} if params is None else params
    tau = params.get('tau', min(1, 1/2 * np.sqrt(kappa_B/model.kappa_G)))
    assert tau >= 0, "The parameter tau must be greater than 0"
    assert tau <= 1, "The parameter tau must be less than 1"
    eta = params.get('eta', 1 / (4*tau*model.L_G))
    theta = params.get('theta', 1 / (eta*L_B))
    alpha = params.get('alpha', model.mu_G)
    assert alpha > 0, "The parameter alpha must be greater than 0"

    # set the initial point
    u = np.zeros(model.dim + model.n * model.m) # for augmented function
    u_f = np.zeros(model.dim + model.n * model.m) # for augmented function
    z = np.zeros(model.n * model.m)
    
    # get CVXPY solution
    x_star, F_star = model.solution
    
    # logging
    x_err = np.zeros(num_steps) # distance
    F_err = np.zeros(num_steps) # function error
    cons_err = np.zeros(num_steps) # constraints error
    primal_dual_err = np.zeros(num_steps) # primal-dual optimality condition error
    
    ts = []
    start = time.time()
    
    for i in range(num_steps):
        u_g = tau * u + (1 - tau) * u_f # point for gradient
        g = model.grad_G(u_g[:model.dim], u_g[model.dim:]) # calculate gradient of the augmented function
        u_half = 1 / (1 + eta * alpha) * (u - eta * (g - alpha * u_g + bB.T @ z)) # half point
        z = z + theta * (bB @ u_half - model.bb)
        u = 1 / (1 + eta * alpha) * (u - eta * (g - alpha * u_g + bB.T @ z)) # next point
        u_prev = u
        u_f = u_g + 2 * tau / (2 - tau) * (u - u_prev) # point for function
        
        end = time.time()
        ts.append(end - start)
        
        # add values to the logs
        x_f = u_f[:model.dim]
        x_err[i] = np.linalg.norm(x_f - x_star)**2 # ||x_f - x*||_2^2
        F_err[i] = abs(model.F(x_f) - F_star) # |F(x_f) - F*|
        cons_err[i] = np.linalg.norm(bB @ u_f - model.bb) # ||K @ xz_f - b||_2
        primal_dual_err[i] = np.linalg.norm(bB.T @ z + model.grad_G(x_f, u_f[model.dim:]))
        
    return x_f, x_err, F_err, cons_err, primal_dual_err, ts


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
        x: float - Solution.
        x_err: float - Distance to the actual solution.
        F_err: float - Function error.
        cons_err: float - Constraints error.
        ts: np.ndarray - Sequence of time taken for previous iterations.
    """

    bB = np.hstack((model.bA, model.gamma * model.bW))
    W = bB.T @ bB
    
    mu_B = utils.get_s2min_plus(bB)
    L_B = utils.get_s2max(bB)
    kappa_B = L_B / mu_B
    
    # set algorithm parameters
    params = {} if params is None else params
    tau = params.get('tau', min(1, 1/2 * np.sqrt(19/(15*model.kappa_G))))
    assert tau >= 0, "The parameter tau must be greater than 0"
    assert tau <= 1, "The parameter tau must be less than 1"
    eta = params.get('eta', 1 / (4*tau*model.L_G))
    theta = params.get('theta', 15 / (19*eta))
    alpha = params.get('alpha', model.mu_G)
    assert alpha > 0, "The parameter alpha must be greater than 0"
    N = int(np.sqrt(kappa_B)) + 1 # can be chosen as N >= sqrt(chi)

    # set the initial point
    #x_f = x = np.zeros(model.dim)
    xz_f = xz = np.zeros(model.dim + model.n * model.m) # for augmented function
    u = np.zeros(model.dim + model.n * model.m)
    
    # get CVXPY solution
    x_star, F_star = model.solution
    
    # logging
    x_err = np.zeros(num_steps) # distance
    F_err = np.zeros(num_steps) # function error
    cons_err = np.zeros(num_steps) # constraints error
    
    ts = []
    start = time.time()
    
    for i in range(num_steps):
        xz_prev = xz # previous point
        xz_g = tau * xz + (1 - tau) * xz_f # point for gradient
        g = model.grad_G(xz_g[:model.dim], xz_g[model.dim:]) # calculate gradient of the augmented function
        xz_half = 1 / (1 + eta * alpha) * (xz - eta * (g - alpha * xz_g + u)) # half point
        r = theta * (xz_half - chebyshev(xz_half, bB, model.bb, N, L_B, mu_B))
        u = u + r
        xz = xz_half - eta / (1 + eta * alpha) * r # next point
        xz_f = xz_g + 2 * tau / (2 - tau) * (xz - xz_prev) # point for function
        
        end = time.time()
        ts.append(end - start)
        
        # add values to the logs
        x_f = xz_f[:model.dim]
        x_err[i] = np.linalg.norm(x_f - x_star)**2 # ||x_f - x*||_2^2
        F_err[i] = abs(model.F(x_f) - F_star) # |F(x_f) - \tilde{F}*|
        cons_err[i] = np.linalg.norm(bB @ xz_f - model.bb) # ||K @ xz_f - b||_2
        
    return x_f, x_err, F_err, cons_err, ts  


###################################################################################################