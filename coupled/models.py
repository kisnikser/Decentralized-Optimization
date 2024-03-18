import numpy as np
import cvxpy as cp
import scipy as sp
from typing import List
import utils

class Model:
    """
    In this class we use notation from the current draft of the paper.
    """
    def __init__(self, 
                 num_nodes: int,
                 num_cons: int,
                 dims: List[int],
                 graph: str = 'ring',
                 edge_prob: float = None,
                 nu: float = None) -> None:
        """
        Args:
            num_nodes: int - Number of nodes in the graph (n).
            num_cons: int - Number of constraints rows (m).
            dims: List[int] - Dimensions of local variables (d_1, ..., d_n).
            graph: str = 'ring' - Network graph model.
            edge_prob: float = None - Probability threshold for Erdos-Renyi graph.
            nu: float = None - Augmentation parameter.
        """
        
        self.n = num_nodes # n
        self.m = num_cons # m
        self.dims = dims # [d_1, ..., d_n]
        self.dim = sum(dims) # [d_1, ..., d_n] -> d_1 + ... + d_n = dim

        # set parameters for constraints
        self.A = [np.random.rand(self.m, self.dims[i]) for i in range(self.n)] # A_1, ..., A_n
        #self.b = [np.random.rand(self.m) for _ in range(self.n)] # b_1, ..., b_n
        self.b = [np.zeros(self.m) for _ in range(self.n)] # b_1, ..., b_n
        
        self.bA = np.hstack(self.A) # bA = [A_1, ..., A_n]
        self.bb = np.sum(self.b, axis=0) # bb = b_1 + ... + b_n
        
        self.bA_prime = sp.linalg.block_diag(*self.A) # bA' = diag(A_1, ..., A_n)
        self.bb_prime = np.hstack(self.b) # bb' = col(b_1, ..., b_n)       
    
        # get Laplacian matrix of the network graph
        if graph == 'ring':
            self.W = utils.get_ring_W(self.n)
        elif graph == 'erdos-renyi':
            self.W = utils.get_ER_W(self.n, edge_prob)
        else:
            raise NotImplementedError
        
        Im = np.identity(self.m) # identity matrix of shape m
        Id = np.identity(self.dim)
        self.bW = np.kron(self.W, Im) # bW = W x Im
        
        self.bB = np.block([[np.identity(self.dim), np.zeros((self.dim, self.n * self.m))],
                            [self.bA_prime, self.bW]])
            
        self._nu = nu # augmentation parameter
        self._mu = None # strongly convexity constant for augmented function
        self._L = None # gradient Lipschitz constant for augmented function
        self._kappa = None # condition number of the augmented function
    
  
    @property
    def nu(self) -> float:
        if self._nu is None:
            self._nu = utils.lambda_min(self.hess_F()) # nu = mu_F
        return self._nu
            
            
    @property
    def mu(self) -> float:
        """
        Returns:
            mu: float - Strongly convexity constant of the augmented function.
        """
        if self._mu is None:
            mu_F = utils.lambda_min(self.hess_F()) # minimum eigenvalue of function F(x)
            mu_Phi = mu_F + self.nu # minimum eigenvalue of function Phi(x, z) >= mu_F + nu
            #_, s, _ = np.linalg.svd(self.bB) # singular values of bB
            #s2min = (s**2)[-1] # squared minimum singular value of bB
            s2min = utils.lambda_min_plus(self.bB.T @ self.bB) # squared minimum singular value of bB
            mu_tildeF = mu_Phi * s2min # mu_tildeF >= mu_Phi * sigma^2_min(bB)
            self._mu = mu_tildeF
        return self._mu
    
         
    @property
    def L(self) -> float:
        """
        Returns:
            L: float - gradient Lipschitz constant of the augmented function.
        """
        if self._L is None:
            L_F = utils.lambda_max(self.hess_F()) # maximum eigenvalue of function F(x)
            L_Phi = L_F + self.nu # maximum eigenvalue of function Phi(x, z) <= L_F + nu
            #_, s, _ = np.linalg.svd(self.bB) # singular values of bB
            #s2max = (s**2)[0] # squared maximum singular value of bB
            s2max = utils.lambda_max(self.bB.T @ self.bB) # squared maximum singular value of bB
            L_tildeF = L_Phi * s2max # L_tildeF <= L_Phi * sigma^2_max(B)
            self._L = L_tildeF
        return self._L
    
    
    @property
    def kappa(self) -> float:
        """
        Returns:
            kappa: float - Condition number of the augmented function.
        """
        if self._kappa is None:
            self._kappa = self.L / self.mu
        return self._kappa
            
    
    def F(self, bx):
        """
        Args:
            bx: np.ndarray - Vector of primal variables.
        Returns:
            Function value at point bx.
        """
        raise NotImplementedError
    
    
    def tildeF(self, bx_bz):
        """
        Args:
            bx_bz: np.ndarray - Vector of stacked primal and additional variables.
        Returns:
            Augmented function value at point (bx, bz).
        """
        bx = bx_bz[:self.dim]
        bz = bx_bz[self.dim:]
        return (
            self.F(bx)
            + self.nu / 2 * np.linalg.norm(self.bA_prime @ bx + self.bW @ bz - self.bb_prime)**2
        )
        
        
    def grad_F(self, bx):
        """
        Args:
            bx: np.ndarray - Vector of primal variables.
        Returns:
            Function gradient at point bx.
        """
        raise NotImplementedError
    

    def grad_tildeF(self, bx_bz):
        """
        Args:
            bx_bz: np.ndarray - Vector of stacked primal and additional variables.
        Returns:
            Augmented function gradient at point (bx, bz).
        """
        bx = bx_bz[:self.dim]
        bz = bx_bz[self.dim:]
        return np.hstack((
            self.grad_F(bx) + self.nu * self.bA_prime.T @ (self.bA_prime @ bx + self.bW @ bz),
            self.nu * self.bW @ (self.bA_prime @ bx + self.bW @ bz)
        ))
    
    
    def hess_F(self, bx: np.ndarray = None):
        """
        Args:
            bx: np.ndarray = None - Vector of primal variables.
        Returns:
            Function hessian at point bx.
        """
        raise NotImplementedError
    
    
    def _split_vector(self, bx):
        """
        Args:
            bx: np.ndarray - Vector of primal variables.
        Returns:
            x = [x_1, ..., x_n]: List[np.ndarray] - List of sub-vectors for each agent.
        """
        split_indices = np.cumsum([0] + self.dims)[:-1]
        x = np.split(bx, split_indices)
        return x
    
    
    def _get_solution(self):
        """
        Returns:
            x.value: Solution.
            prob.value: Function value at solution.
        """
        raise NotImplementedError