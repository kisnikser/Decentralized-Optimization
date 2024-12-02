import numpy as np
import cvxpy as cp
import scipy as sp
from models import Model, ExampleModel, VFL
from typing import Dict, Optional
import utils
import time

from utils import Timer

from tqdm import tqdm

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
        output: Dict[...] - Dictionary with the method results
    """
    
    if model.model_type == 'ExampleModel':
        get_argmin_TrackingADMM = get_argmin_TrackingADMM_example
    elif model.model_type == 'VFL':
        get_argmin_TrackingADMM = get_argmin_TrackingADMM_vfl
    else:
        raise NotImplementedError
    
    # set variables to the paper notation
    #W = np.kron(model.mixing_matrix, np.identity(model.m))
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
    
    grad_calls = []
    mults_A = []
    communications = []
    
    if model.model_type == 'ExampleModel':
        A = model.bCT_bC + model.theta * np.identity(model.dim) + c * A_d.T @ A_d # +2 mult A, A^T
    elif model.model_type == 'VFL':
        A = model.hess_F() + c * A_d.T @ A_d # +2 mult A, A^T
    #mults_A.append(2)
    
    for i in tqdm(range(num_steps)):
        # algorithm step
        x_prev = x
        #with Timer('get_argmin_TrackingADMM'):
        x, next_grad_calls = get_argmin_TrackingADMM(x_prev, d, lmbd, c, model, A) # + 3 mult A, A^T, +2 communication
        d = (model.W @ d.reshape((model.n, model.m))).flatten() + A_d @ (x - x_prev) # +1 communication, +1 mult A
        lmbd = (model.W @ lmbd.reshape((model.n, model.m))).flatten() + c * d # +1 communication
       
        # time 
        end = time.time()
        ts.append(end - start)
        
        # oracles
        grad_calls.append(next_grad_calls)
        mults_A.append(4)
        communications.append(4)
        
        # add values to the logs
        x_err[i] = np.linalg.norm(x - x_star)**2 # ||x - x*||_2^2
        F_err[i] = abs(model.F(x) - F_star) # |F(x) - F*|
        cons_err[i] = np.linalg.norm(model.A_hstacked @ x - model.b_sum) # ||bA @ x - bb||_2
        
    output = {
        'x': x,
        'x_err': x_err,
        'F_err': F_err,
        'cons_err': cons_err,
        'ts': ts,
        'grad_calls': grad_calls,
        'mults_A': mults_A,
        'communications': communications
    }
        
    return output

#--------------------------------------------------------------------------------------------------

# Example model
def get_argmin_TrackingADMM_example(x_k: np.ndarray,
                                    d_k: np.ndarray,
                                    lmbd_k: np.ndarray,
                                    c: float,
                                    model: ExampleModel,
                                    A: np.ndarray):
    """
    Solve argmin subproblem in Tracking-ADMM for our Example problem.
    
    Args:
        x_k: np.ndarray - Value of primal variable vector x from previous step.
        d_k: np.ndarray - Value of vector d from previous step.
        lmbd_k: np.ndarray - Value of vector lmbd from previous step.
        c: float - Parameter for augmentation.
        model: VFL - Model with oracle, which gives F, grad_F, etc.
    Returns:
        x: np.ndarray - Solution
        grad_calls: int - Number of gradient calls during the algorithm
    """
    # set variables to the paper notation
    #W = np.kron(model.mixing_matrix, np.identity(model.m))
    A_d = model.bA
    
    b = (
        model.bCT_bd
        - A_d.T @ (model.W @ lmbd_k.reshape((model.n, model.m))).flatten() # +1 mult A^T, +1 communication
        + c * A_d.T @ (A_d @ x_k - (model.W @ d_k.reshape((model.n, model.m))).flatten()) # +2 mult A, A^T, +1 communication
    )
    
    x, grad_calls = ConjugateGradientQuadratic(x_k, A, b)
        
    return x, grad_calls

#--------------------------------------------------------------------------------------------------

# VFL
def get_argmin_TrackingADMM_vfl(x_k: np.ndarray,
                                d_k: np.ndarray,
                                lmbd_k: np.ndarray,
                                c: float,
                                model: VFL,
                                A: np.ndarray):
    """
    Solve argmin subproblem in Tracking-ADMM for our VFL problem.
    
    Args:
        x_k: np.ndarray - Value of primal variable vector x from previous step.
        d_k: np.ndarray - Value of vector d from previous step.
        lmbd_k: np.ndarray - Value of vector lmbd from previous step.
        c: float - Parameter for augmentation.
        model: VFL - Model with oracle, which gives F, grad_F, etc.
        mode: str = 'newton' - Use newton or CVXPY.
    Returns:
        x: np.ndarray - Solution
        grad_calls: int - Number of gradient calls during the algorithm
    """
    # set variables to the paper notation
    #W = np.kron(model.mixing_matrix, np.identity(model.m))
    A_d = model.bA
    
    b = (
        model._rearrange_vector(np.hstack((model.l, np.zeros(model.n * model.d))))
        - A_d.T @ (model.W @ lmbd_k.reshape((model.n, model.m))).flatten() # +1 mult A^T, +1 communication
        + c * A_d.T @ (A_d @ x_k - (model.W @ d_k.reshape((model.n, model.m))).flatten()) # +2 mult A, A^T, +1 communication
    )
    
    x, grad_calls = ConjugateGradientQuadratic(x_k, A, b)
    
    return x, grad_calls


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
        output: Dict[...] - Dictionary with the method results
    """
    
    if model.model_type == 'ExampleModel':
        get_argmin_DPMM = get_argmin_DPMM_example
    elif model.model_type == 'VFL':
        get_argmin_DPMM = get_argmin_DPMM_vfl
    else:
        raise NotImplementedError
    
    # set variables to the paper notation
    I_n = np.identity(model.dim)
    #bL = model.bW
    G_d = lambda x: model.bA @ x - model.bb
    
    # set algorithm parameters
    params = {} if params is None else params
    
    theta = params.get('theta', np.ones(model.n))
    assert np.all(theta > 0) and np.all(theta < 2), "Parameter theta must be greater than 0 and less than 2"
    Theta = sp.linalg.block_diag(*[theta[i] * np.identity(model.dimensions[i]) for i in range(model.n)])
    
    alpha = params.get('alpha', np.ones(model.n))
    assert np.all(alpha > 0), "Parameter alpha must be greater than 0"
    Upsilon = sp.linalg.block_diag(*[alpha[i] * np.identity(model.dimensions[i]) for i in range(model.n)])
    
    gamma = params.get('gamma', np.ones(model.n))
    assert np.all(gamma > 0), "Parameter gamma must be greater than 0"
    Gamma = sp.linalg.block_diag(*[gamma[i] * np.identity(model.m) for i in range(model.n)])
    
    beta = params.get('beta', min(1 / (gamma * utils.lambda_max(model.W))) / 2)
    assert beta > 0 and beta < min(1 / (gamma * utils.lambda_max(model.W))), "Wrong parameter beta"
    
    #beta = params.get('beta', min(1 / (gamma * utils.lambda_max(bL))) / 2)
    #assert beta > 0 and beta < min(1 / (gamma * utils.lambda_max(bL))), "Wrong parameter beta"
    
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
    
    grad_calls = []
    mults_A = []
    communications = []
    
    A = (
        model.hess_F()
        + model.bA.T @ Gamma @ model.bA # + 2 mult A, A^T
        + np.linalg.inv(Upsilon)
    )
    #mults_A.append(2)
    
    for i in tqdm(range(num_steps)):
        # algorithm step
        #with Timer('get_argmin_DPMM'):
        x_hat, next_grad_calls = get_argmin_DPMM(x, y - Gamma @ Lambda, Gamma, Upsilon, model, A) # +1 mult A
        y_hat = y - Gamma @ Lambda + Gamma @ G_d(x_hat) # +1 mult A 
        x = (I_n - Theta) @ x + Theta @ x_hat
        Lambda_prev = Lambda
        #Lambda = Lambda_prev + beta * bL @ y_hat # +1 communication
        #bL @ y_hat заменяется на:
        bL_times_yhat = (model.W @ y_hat.reshape((model.n, model.m))).flatten()
        Lambda = Lambda_prev + beta * bL_times_yhat # +1 communication
        y = y_hat + Gamma @ (Lambda_prev - Lambda)
        
        # time
        end = time.time()
        ts.append(end - start)
        
        # oracles
        grad_calls.append(next_grad_calls)
        mults_A.append(2)
        communications.append(1)
        
        # add values to the logs
        x_err[i] = np.linalg.norm(x - x_star)**2 # ||x - x*||_2^2
        F_err[i] = abs(model.F(x) - F_star) # |F(x) - F*|
        cons_err[i] = np.linalg.norm(model.A_hstacked @ x - model.b_sum) # ||bA @ x - bb||_2
    
    output = {
        'x': x,
        'x_err': x_err,
        'F_err': F_err,
        'cons_err': cons_err,
        'ts': ts,
        'grad_calls': grad_calls,
        'mults_A': mults_A,
        'communications': communications
    }
        
    return output

#--------------------------------------------------------------------------------------------------

# Example model
def get_argmin_DPMM_example(x_k: np.ndarray,
                            y: np.ndarray,
                            Gamma: np.ndarray,
                            Upsilon: np.ndarray,
                            model: ExampleModel,
                            A: np.ndarray):
    """
    Solve argmin subproblem in DPMM for our Example problem.
    
    Args:
        x_k: np.ndarray - Value of primal variable vector x from previous step.
        y: np.ndarray - Value of vector y from previous step.
        Gamma: np.ndarray - Block matrix with gamma (see notation).
        Upsilon: np.ndarray - Block matrix with alpha (see notation).
        model: ExampleModel - Model with oracle, which gives F, grad_F, etc.
    Returns:
        x: np.ndarray - Solution
        grad_calls: int - Number of gradient calls during the algorithm
    """
    
    b = (
        model.bCT_bd
        + model.bA.T @ (Gamma @ model.bb - y) # + 1 mult A^T
        + np.linalg.inv(Upsilon) @ x_k
    )
    
    #with Timer('ConjugateGradientQuadratic'):
    x, grad_calls = ConjugateGradientQuadratic(x_k, A, b)
    
    return x, grad_calls

#--------------------------------------------------------------------------------------------------

# VFL
def get_argmin_DPMM_vfl(x_k: np.ndarray,
                        y: np.ndarray,
                        Gamma: np.ndarray,
                        Upsilon: np.ndarray,
                        model: VFL,
                        A: np.ndarray):
    """
    Solve argmin subproblem in DPMM for our VFL problem.
    
    Args:
        x_k: np.ndarray - Value of primal variable vector x from previous step.
        y: np.ndarray - Value of vector y from previous step.
        alpha: np.ndarray - Vector of parameters alpha.
        gamma: np.ndarray - Vector of parameters gamma.
        model: ExampleModel - Model with oracle, which gives F, grad_F, etc.
    Returns:
        x: np.ndarray - Solution
        grad_calls: int - Number of gradient calls during the algorithm
    """
    
    b = (
        model._rearrange_vector(np.hstack((model.l, np.zeros(model.n * model.d))))
        + model.bA.T @ (Gamma @ model.bb - y) # + 1 mult A^T
        + np.linalg.inv(Upsilon) @ x_k
    )
    
    #with Timer('ConjugateGradientQuadratic'):
    x, grad_calls = ConjugateGradientQuadratic(x_k, A, b)
    
    return x, grad_calls


###################################################################################################


def ConjugateGradientQuadratic(x0, A, b, tol=1e-8):
    """
    Realization of Conjugate Gradients for quadratic function 1/2 x^T A x - b^T x = 0.
    Args:
        x0: np.ndarray - Initial point
        A: np.ndarray - Matrix of quadratic form
        b: np.ndarray - Vector for linear term
    Returns:
        x: np.ndarray - Solution
        grad_calls: int - Number of gradient calls during the algorithm
    """
    max_iter = len(x0)
    x = x0
    r = A.dot(x0) - b # +1 grad call
    grad_calls = 1
    p = -r
    for _ in range(max_iter):
        Ap = A.dot(p)
        alpha = r.dot(r) / p.dot(Ap)
        x = x + alpha * p
        r_next = r + alpha * Ap # +1 grad call
        grad_calls += 1
        beta = r_next.dot(r_next) / r.dot(r)
        p = -r_next + beta * p
        r = r_next
        if np.linalg.norm(r) < tol:
            break
    return x, grad_calls


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
        output: Dict[...] - Dictionary with the method results
    """
    
    K = model.bB
    mu_K = model.mu_B
    L_K = model.L_B
    kappa_K = model.kappa_B
    
    # set algorithm parameters
    params = {} if params is None else params
    tau = params.get('tau', min(1, 1/2 * np.sqrt(kappa_K/model.kappa_G)))
    assert tau >= 0, "The parameter tau must be greater than 0"
    assert tau <= 1, "The parameter tau must be less than 1"
    eta = params.get('eta', 1 / (4*tau*model.L_G))
    theta = params.get('theta', 1 / (eta*L_K))
    alpha = params.get('alpha', model.mu_G)
    assert alpha > 0, "The parameter alpha must be greater than 0"

    # set the initial point
    u = np.zeros(model.dim + model.n * model.m) # for augmented function
    u_f = np.zeros(model.dim + model.n * model.m) # for augmented function
    z = np.zeros(model.dim + model.n * model.m)
    
    # get CVXPY solution
    x_star, F_star = model.solution
    
    # logging
    x_err = np.zeros(num_steps) # distance
    F_err = np.zeros(num_steps) # function error
    cons_err = np.zeros(num_steps) # constraints error
    #primal_dual_err = np.zeros(num_steps) # primal-dual optimality condition error
    
    ts = []
    start = time.time()
    
    grad_calls = []
    mults_A = []
    communications = []
    
    for i in tqdm(range(num_steps)):
        u_g = tau * u + (1 - tau) * u_f 
        g = model.grad_G(u_g[:model.dim], u_g[model.dim:]) # +1 grad call, +2 mult A, A^T, +2 communication
        u_half = 1 / (1 + eta * alpha) * (u - eta * (g - alpha * u_g + z)) # +1 mult A^T, +1 communication
        z = z + theta * K.T @ (K @ u_half - model.bb) # +1 mult A^T, +1 communication
        u_prev = u
        u = 1 / (1 + eta * alpha) * (u - eta * (g - alpha * u_g + z)) # +1 mult A^T, +1 communication
        u_f = u_g + 2 * tau / (2 - tau) * (u - u_prev)
        
        # time
        end = time.time()
        ts.append(end - start)
        
        # oracles
        grad_calls.append(1)
        mults_A.append(5)
        communications.append(5)
        
        # add values to the logs
        x_f = u_f[:model.dim]
        x_err[i] = np.linalg.norm(x_f - x_star)**2
        F_err[i] = abs(model.F(x_f) - F_star)
        cons_err[i] = np.linalg.norm(K @ u_f - model.bb)
        #primal_dual_err[i] = np.linalg.norm(K.T @ z + model.grad_G(x_f, u_f[model.dim:]))

    output = {
        'x': x_f,
        'x_err': x_err,
        'F_err': F_err,
        'cons_err': cons_err,
        #'primal_dual_err': primal_dual_err,
        'ts': ts,
        'grad_calls': grad_calls,
        'mults_A': mults_A,
        'communications': communications
    }
            
    return output


###################################################################################################


def Chebyshev(v, M, r):
    """
    Chebyshev iteration.
    
    Args:
        v: np.ndarray - Vector
        M: np.ndarray - Matrix
        r: np.ndarray - Vector from the range of M
    Returns:
        v^n: np.ndarray - Vector
    """
    
    L_M = utils.get_s2max(M)
    mu_M = utils.get_s2min_plus(M)
    n = np.ceil(np.sqrt(L_M / mu_M)).astype(int)
    
    rho = (L_M - mu_M)**2 / 16
    nu = (L_M + mu_M) / 2
    
    delta = - nu / 2
    p = - M.T @ (M @ v - r) / nu
    v = v + p
        
    for _ in range(1, n):
        beta = rho / delta
        delta = - (nu + beta)
        p = (M.T @ (M @ v - r) + beta * p) / delta
        v = v + p
            
    return v


###################################################################################################


def mulW_prime(y, model):
    """
    Multiplication by W'.
    
    Args:
        y: np.ndarray - Vector
        model: Model - Model with bW, L_W, mu_W, kappa_W
    Returns:
        W'y: np.ndarray - Multiplication by W'
    """
    
    n = np.ceil(np.sqrt(model.kappa_W)).astype(int)
    
    rho = (np.sqrt(model.L_W) - np.sqrt(model.mu_W))**2 / 16
    nu = (np.sqrt(model.L_W) + np.sqrt(model.mu_W)) / 2
    
    delta = - nu / 2
    #p = - model.bW @ y / nu
    p = - (model.W @ y.reshape((model.n, model.m))).flatten() / nu
    y_0 = y.copy()
    y = y + p
        
    for _ in range(1, n):
        beta = rho / delta
        delta = - (nu + beta)
        p = ((model.W @ y.reshape((model.n, model.m))).flatten() + beta * p) / delta
        y = y + p
            
    return y_0 - y


###################################################################################################


def K_Chebyshev(u, model):
    """
    Computation of K.T @ (K @ u - b').
    
    Args:
        u: np.ndarray - Vector of stacked x and y, i.e. u = (x, y)
    Returns:
        K.T @ (K @ u - b'): np.ndarray - Resulting vector for this multiplication
    """
    
    x, y = u[:model.dim], u[model.dim:]
    
    mu_W_prime = (11 / 15) ** 2
    L_W_prime = (19 / 15) ** 2
    
    mu_B = model.mu_A / 2
    L_B = model.L_A + (model.L_A + model.mu_A) * L_W_prime / mu_W_prime
    kappa_B = L_B / mu_B
    gamma = np.sqrt((model.mu_A + model.L_A) / mu_W_prime)
    
    n = np.ceil(np.sqrt(kappa_B)).astype(int)
    
    rho = (L_B - mu_B)**2 / 16
    nu = (L_B + mu_B) / 2
    
    delta = - nu / 2
    q = model.bA @ x + gamma * mulW_prime(y, model) - model.bb # +1 mult A, +n(W) communications
    p = - 1 / nu * np.hstack((model.bA.T @ q, gamma * mulW_prime(q, model))) # +1 mult A^T, +n(W) communications
    u_0 = u.copy()
    u = u_0 + p
    
    for _ in range(1, n):
        beta = rho / delta
        delta = - (nu + beta)
        x, y = u[:model.dim], u[model.dim:]
        q = model.bA @ x + gamma * mulW_prime(y, model) - model.bb # +1 mult A, +n(W) communications
        p = 1 / delta * np.hstack((model.bA.T @ q, gamma * mulW_prime(q, model))) + beta * p / delta # +1 mult A^T, +n(W) communications
        u = u + p
        
    return u_0 - u


###################################################################################################


def grad_G(u, model):
        """
        Args:
            u: np.ndarray - Vector of stacked variables u = (x, y)
            model: Model - Model
        Returns:
            Augmented function gradient at point (x, y).
        """
        mu_W_prime = (11 / 15) ** 2
        gamma = np.sqrt((model.mu_A + model.L_A) / mu_W_prime)
        
        x, y = u[:model.dim], u[model.dim:]
        z = model.r * (model.bA @ x + gamma * mulW_prime(y, model) - model.bb) # +1 mult A, +n(W) communications
        return np.hstack((
            model.grad_F(x) + model.bA.T @ z,  # +1 grad call, +1 mult A^T
            gamma * mulW_prime(z, model) # +n communications
        ))

###################################################################################################


def Main(num_steps: int, 
        model: Model, 
        params: Dict[str, float] = None):
    """
    Our main algoritm (see Algorithm 6 in the paper).
    
    Args:
        num_steps: int - Number of optimizer steps.
        model: Model - Model with oracle, which gives F, grad_F, etc.
        params: Dict[str, float] = None - Algorithm parameters.
    Returns:
        output: Dict[...] - Dictionary with the method results
    """
    
    mu_K = 11 / 15
    L_K = 19 / 15
    kappa_K = L_K / mu_K
    
    mu_W_prime = (11 / 15) ** 2
    L_W_prime = (19 / 15) ** 2
    kappa_W_prime = L_W_prime / mu_W_prime
    
    mu_G = model.mu_f * min(1 / 2, (model.mu_A + model.L_A) / (4 * model.L_A))
    L_G = max(model.L_f + model.mu_f, model.mu_f * (model.mu_A + model.L_A) / model.L_A * L_W_prime / mu_W_prime)
    kappa_G = L_G / mu_G
    
    mu_B = model.mu_A / 2
    L_B = model.L_A + (model.L_A + model.mu_A) * L_W_prime / mu_W_prime
    kappa_B = L_B / mu_B
    
    # set algorithm parameters
    params = {} if params is None else params
    tau = params.get('tau', min(1, 1/2 * np.sqrt(kappa_K/kappa_G)))
    assert tau >= 0, "The parameter tau must be greater than 0"
    assert tau <= 1, "The parameter tau must be less than 1"
    eta = params.get('eta', 1 / (4*tau*L_G))
    theta = params.get('theta', 1 / (eta*L_K))
    alpha = params.get('alpha', mu_G)
    assert alpha > 0, "The parameter alpha must be greater than 0"

    # set the initial point
    u = np.zeros(model.dim + model.n * model.m) # for augmented function
    u_f = np.zeros(model.dim + model.n * model.m) # for augmented function
    z = np.zeros(model.dim + model.n * model.m)
    
    # get CVXPY solution
    x_star, F_star = model.solution
    
    # logging
    x_err = np.zeros(num_steps) # distance
    F_err = np.zeros(num_steps) # function error
    cons_err = np.zeros(num_steps) # constraints error
    #primal_dual_err = np.zeros(num_steps) # primal-dual optimality condition error
    
    ts = []
    start = time.time()
    
    grad_calls = []
    mults_A = []
    communications = []
    
    for i in tqdm(range(num_steps)):
        u_g = tau * u + (1 - tau) * u_f 
        g = grad_G(u_g, model) # +1 grad call, +2 mult A, A^T, +2n(W) communications
        u_half = 1 / (1 + eta * alpha) * (u - eta * (g - alpha * u_g + z))
        #with Timer('K_Chebyshev'):
        z = z + theta * K_Chebyshev(u_half, model) # +2n(B) mult A, A^T, +2n(W)n(B) communications
        u_prev = u
        u = 1 / (1 + eta * alpha) * (u - eta * (g - alpha * u_g + z))
        u_f = u_g + 2 * tau / (2 - tau) * (u - u_prev)
        
        # time
        end = time.time()
        ts.append(end - start)
        
        # oracles
        grad_calls.append(1)
        mults_A.append(2 + 2 * np.ceil(np.sqrt(kappa_B)).astype(int))
        communications.append(
            2 * np.ceil(np.sqrt(kappa_W_prime)).astype(int) * (np.ceil(np.sqrt(kappa_B)).astype(int) + 1)
        )
        
        # add values to the logs
        x_f = u_f[:model.dim]
        x_err[i] = np.linalg.norm(x_f - x_star)**2
        F_err[i] = abs(model.F(x_f) - F_star)
        #cons_err[i] = np.linalg.norm(model.bB @ u_f - model.bb) 
        #primal_dual_err[i] = np.linalg.norm(K.T @ z + model.grad_G(x_f, u_f[model.dim:]))

    output = {
        'x': x_f,
        'x_err': x_err,
        'F_err': F_err,
        #'cons_err': cons_err,
        #'primal_dual_err': primal_dual_err,
        'ts': ts,
        'grad_calls': grad_calls,
        'mults_A': mults_A,
        'communications': communications
    }
            
    return output


###################################################################################################


def EXTRA(num_steps: int, 
        model: Model, 
        params: Dict[str, float] = None):
    """
    Our main algoritm (see Algorithm 6 in the paper).
    
    Args:
        num_steps: int - Number of optimizer steps.
        model: Model - Model with oracle, which gives F, grad_F, etc.
        params: Dict[str, float] = None - Algorithm parameters.
    Returns:
        output: Dict[...] - Dictionary with the method results
    """
    
    I_n = np.identity(model.n)
    # W = model.mixing_matrix
    W = model.mixing_matrix
    W_tilde = 0.5 * (I_n + W)
    
    # set algorithm parameters
    params = {} if params is None else params
    alpha = params.get('alpha', utils.lambda_min(W_tilde) / model.L_f)
    assert alpha > 0, "The parameter alpha must be greater than 0"
    assert alpha < 2 * utils.lambda_min(W_tilde) / model.L_f

    # alpha = 1e-5

    # set the initial point
    x = np.zeros((model.n, model.d))
    
    # get CVXPY solution
    x_star, F_star = model.solution
    
    # logging
    x_err = np.zeros(num_steps) # distance
    F_err = np.zeros(num_steps) # function error
    cons_err = np.zeros(num_steps) # constraints error
    #primal_dual_err = np.zeros(num_steps) # primal-dual optimality condition error
    
    ts = []
    start = time.time()
    
    grad_calls = []
    mults_A = []
    communications = []
    
    def grad_F_EXTRA(x):
        return model.grad_F(x.reshape(-1)).reshape((model.n, model.d))
    
    x_prev = x
    x = W @ x_prev - alpha * grad_F_EXTRA(x_prev)
    
    for i in tqdm(range(num_steps)):
        
        x, x_prev = (I_n + W) @ x - W_tilde @ x_prev - alpha * (grad_F_EXTRA(x) - grad_F_EXTRA(x_prev)), x
        
        # time
        end = time.time()
        ts.append(end - start)
        
        # oracles
        grad_calls.append(1)
        mults_A.append(1) # ???
        communications.append(1)
        
        # add values to the logs
        x_f = x.reshape(-1)
        x_err[i] = np.linalg.norm(x_f - x_star)**2
        F_err[i] = abs(model.F(x_f) - F_star)

    output = {
        'x': x_f,
        'x_err': x_err,
        'F_err': F_err,
        #'cons_err': cons_err,
        #'primal_dual_err': primal_dual_err,
        'ts': ts,
        'grad_calls': grad_calls,
        'mults_A': mults_A,
        'communications': communications
    }
            
    return output