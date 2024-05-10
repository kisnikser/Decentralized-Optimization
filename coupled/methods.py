import numpy as np
import cvxpy as cp
import scipy as sp
from models import Model, ExampleModel, VFL
from typing import Dict
import utils


###################################################################################################


def algorithm_1(num_steps: int,
                model: Model,
                params: Dict[str, float] = None):
    """
    Algorithm 1 from the our paper.
    
    Args:
        num_steps: int - Number of optimizer steps.
        model: NewModel - NewModel with oracle, which gives F, grad_F, etc.
        params: Dict[str, float] = None - Algorithm parameters.
    Returns:
        x: float - Solution.
        x_err: np.ndarray - Sequence of distances to the actual solution.
        F_err: np.ndarray - Sequence of function error.
        cons_err: np.ndarray - Sequence of constraints error.
        primal_dual_err: np.ndarray - Sequence of primal-dual optimality condition error.
    """

    # set algorithm parameters
    params = {} if params is None else params
    
    gamma_x = params.get('gamma_x', 1 / (2 * max(model.L_f + model.mu_f, 2 * model.mu_f * model.kappa_W)))
    gamma_y = params.get('gamma_y', gamma_x * (model.mu_A + model.L_A) / model.mu_W)
    
    eta_x = params.get('eta_x', min(1 / (np.sqrt(2 * (model.mu_f**2 + model.mu_f * model.L_f))),
                                    np.sqrt(gamma_x / (8 * model.mu_f * model.kappa_A))))
    eta_y = params.get('eta_y', min(model.L_A / (model.mu_f * np.sqrt(model.mu_W * model.L_W)),
                                    np.sqrt(gamma_x * model.mu_A * model.L_A / (4 * model.mu_f * model.mu_W * model.L_W))))
    eta_z = params.get('eta_z', min(1 / (6 * gamma_x * model.kappa_W * model.L_A),
                                    1 / (16 * eta_x * model.L_A),
                                    1 / (16 * eta_y * model.L_W)))
    
    alpha = params.get('alpha', min(1 / 2,
                                    eta_x * model.mu_f / 2,
                                    eta_y * model.mu_f * model.mu_W / (4 * model.L_A)))
    beta = params.get('beta', max(1 - alpha / 2,
                                  1 - eta_z * gamma_x * model.mu_A / 2))

    # set the initial point
    x = np.zeros(model.dim)
    x_bar = np.zeros(model.dim)
    
    y = np.zeros(model.n * model.m)
    y_bar = np.zeros(model.n * model.m)
    
    z = np.zeros(model.n * model.m)
    z_prev = np.zeros(model.n * model.m)
    
    # get CVXPY solution
    x_star, F_star = model.solution
    #_, w_star = model._split_vector(x_star)

    # logging
    x_err = np.zeros(num_steps) # distance
    F_err = np.zeros(num_steps) # function error
    cons_err = np.zeros(num_steps) # constraints error
    primal_dual_err = np.zeros(num_steps) # primal-dual optimality condition error
    
    for i in range(num_steps):
        
        x_und = alpha * x + (1 - alpha) * x_bar
        y_und = alpha * y + (1 - alpha) * y_bar
        z_hat = z + beta * (z - z_prev)

        g_x = model.grad_G_x(x_und, y_und)
        g_y = model.grad_G_y(x_und, y_und)
        x_next = x + alpha * (x_und - x) - eta_x * (g_x - model.bA.T @ z_hat)
        y_next = y + alpha * (y_und - y) - eta_y * model.W_times_I @ (g_y - z_hat)
        
        g_x = model.grad_G_x(x_bar, y_bar)
        g_y = model.grad_G_y(x_bar, y_bar)
        
        x_bar = x_und + alpha * (x_next - x)
        y_bar = y_und + alpha * (y_next - y)
        
        delta_x = model.bA @ (model.bA.T @ z - g_x)
        delta_y = model.W_times_I @ (z - g_y)
        
        z = z - eta_z * (model.bA @ x_next + y_next - model.bb + gamma_x * delta_x + gamma_y * delta_y)
        
        x = x_next
        y = y_next
        
        # add values to the logs
        #_, w = model._split_vector(x)
        #x_err[i] = np.linalg.norm(w - w_star)**2 # ||x - x*||_2^2
        x_err[i] = np.linalg.norm(x - x_star)**2 # ||x - x*||_2^2
        F_err[i] = abs(model.F(x) - F_star) # |F(x) - F*|
        cons_err[i] = np.linalg.norm(model.A_hstacked @ x - model.b_sum) # ||bA @ x - bb||_2
        primal_dual_err[i] = np.linalg.norm(model.grad_G_x(x, y) - model.bA.T @ z) # |||gradG_x(x, y) - bA.T @ z|_2
        
    return x, x_err, F_err, cons_err, primal_dual_err


###################################################################################################


def DPMM(num_steps: int, 
         model: Model, 
         params: Dict[str, float] = None):
    """
    Decentralized Proximal Method of Multipliers (DPMM) from the paper
    "Decentralized Proximal Method of Multipliers for Convex Optimization with Coupled Constraints", 2023.
    
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
    
    if isinstance(model, ExampleModel):
        get_argmin_DPMM = get_argmin_DPMM_example
    elif isinstance(model, VFL):
        get_argmin_DPMM = get_argmin_DPMM_vfl
    else:
        raise NotImplementedError
    
    # set variables to the paper notation
    I_n = np.identity(model.dim)
    bL = model.W_times_I
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
        cons_err[i] = np.linalg.norm(model.A_hstacked @ x - model.b_sum) # ||bA @ x - bb||_2
        
    return x, x_err, F_err, cons_err

#--------------------------------------------------------------------------------------------------

# Example model
def get_argmin_DPMM_example(x_k: np.ndarray,
                            y: np.ndarray,
                            alpha: np.ndarray,
                            gamma: np.ndarray,
                            model: ExampleModel,
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
        model: ExampleModel - Model with oracle, which gives F, grad_F, etc.
        mode: str = 'newton' - Use newton or CVXPY.
    Returns:
        sol: np.ndarray - Solution for argmin subproblem.
    """
    x_k_array = np.split(x_k, np.cumsum([model.d for _ in range(model.n)])[:-1])
    y_array = np.split(y, np.cumsum([model.m for _ in range(model.n)])[:-1])
    sol = []

    if mode == 'newton':
        for i in range(model.n):
            # calculate gradient function
            grad = lambda x: (
                model.C[i].T @ model.C[i] @ x - model.C[i].T @ model.d_[i] + model.theta * x
                + model.A[i].T @ (y_array[i] + gamma[i] * (model.A[i] @ x - model.b[i]))
                + 1 / alpha[i] * (x - x_k_array[i])
            )
            
            # calculate hessian matrix
            hess = lambda x: (
                model.C[i].T @ model.C[i] + model.theta * np.identity(model.d)
                + gamma[i] * model.A[i].T @ model.A[i]
                + 1 / alpha[i] * np.identity(model.d)
            ) 
            
            # get a solution by one newton step
            sol.extend(x_k_array[i] - np.linalg.inv(hess(x_k_array[i])) @ grad(x_k_array[i]))
            
        sol = np.array(sol)
        
    else:
        raise NotImplementedError
    
    return sol

#--------------------------------------------------------------------------------------------------

# VFL
def get_argmin_DPMM_vfl(x_k: np.ndarray,
                        y: np.ndarray,
                        alpha: np.ndarray,
                        gamma: np.ndarray,
                        model: VFL,
                        mode: str = 'newton'):
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
        mode: str = 'newton' - Use newton or CVXPY.
    Returns:
        sol: np.ndarray - Solution for argmin subproblem.
    """
        
    cumsum = np.cumsum(model.dimensions)[:-1]
    x_k_array = np.split(x_k, cumsum)
    
    if model.labels_distribution:
        cumsum = np.cumsum(model.num_samples)[:-1]
        l_array = np.split(model.l, cumsum)
    else:
        l_array = [model.l]
        
    y_array = np.split(y, np.cumsum([model.m for _ in range(model.n)])[:-1])
    sol = []

    if mode == 'newton':
        for i in range(model.n):
            
            # calculate gradient function
            def grad(x):
                
                if model.labels_distribution or i == 0:
                    w = x[:model.d]
                    z = x[model.d:]
                    grad_f = np.hstack((2 * model.lmbd * w, z - l_array[i]))    
                else:
                    grad_f = 2 * model.lmbd * x
                    
                return (
                    grad_f
                    + model.A[i].T @ (y_array[i] + gamma[i] * (model.A[i] @ x - model.b[i]))
                    + 1 / alpha[i] * (x - x_k_array[i])
                )
                
            
            # calculate hessian matrix
            def hess(x):
                
                if model.labels_distribution or i == 0:
                    w = x[:model.d]
                    z = x[model.d:]
                    hess_f = sp.linalg.block_diag(2 * model.lmbd * np.identity(model.d),
                                                  np.identity(model.dimensions[i] - model.d))    
                else:
                    hess_f = 2 * model.lmbd * np.identity(model.dimensions[i])
            
                return (
                    hess_f
                    + gamma[i] * model.A[i].T @ model.A[i]
                    + 1 / alpha[i] * np.identity(model.dimensions[i])
                ) 
            
            # get a solution by one newton step
            sol.extend(x_k_array[i] - np.linalg.inv(hess(x_k_array[i])) @ grad(x_k_array[i]))
            
        sol = np.array(sol)
        
    else:
        raise NotImplementedError
    
    return sol


###################################################################################################


def TrackingADMM(num_steps: int, 
                 model: Model, 
                 params: Dict[str, float] = None):
    """
    Tracking-ADMM algorithm from the paper
    "Tracking-ADMM for distributed constraint-coupled optimization", 2020.
    
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
    c = params.get('c', 1e-3)
    assert c > 0, "Parameter c must be greater than 0"
    
    # set the initial point
    x = np.zeros(model.dim)
    d = np.zeros(model.n * model.m)
    lmbd = np.zeros(model.n * model.m)
    
    # get CVXPY solution
    x_star, F_star = model.solution
    
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
        cons_err[i] = np.linalg.norm(model.A_hstacked @ x - model.b_sum) # ||bA @ x - bb||_2
        
    return x, x_err, F_err, cons_err  

#--------------------------------------------------------------------------------------------------

# Example model
def get_argmin_TrackingADMM_example(x_k: np.ndarray,
                                    d_k: np.ndarray,
                                    lmbd_k: np.ndarray,
                                    c: float,
                                    model: ExampleModel,
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
        model: VFL - Model with oracle, which gives F, grad_F, etc.
        mode: str = 'newton' - Use newton or CVXPY.
    Returns:
        sol: np.ndarray - Solution for argmin subproblem.
    """
    # set variables to the paper notation
    W = np.kron(model.mixing_matrix, np.identity(model.m))
    A_d = model.bA
    
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

###################################################################################################