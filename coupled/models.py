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
                 nu: float = None,
                 gossip: bool = False) -> None:
        """
        Args:
            num_nodes: int - Number of nodes in the graph (n).
            num_cons: int - Number of constraints rows (m).
            dims: List[int] - Dimensions of local variables (d_1, ..., d_n).
            graph: str = 'ring' - Network graph model.
            edge_prob: float = None - Probability threshold for Erdos-Renyi graph.
            nu: float = None - Augmentation parameter.
            gossip: bool = False - W will be gossip matrix defined by metropolis weights.
        """
        
        self.n = num_nodes # n
        self.m = num_cons # m
        self.dims = dims # [d_1, ..., d_n]
        self.dim = sum(dims) # [d_1, ..., d_n] -> d_1 + ... + d_n = dim

        # set parameters for constraints
        self.A = [np.random.rand(self.m, self.dims[i]) for i in range(self.n)] # A_1, ..., A_n
        self.b = [np.random.rand(self.m) for _ in range(self.n)] # b_1, ..., b_n
        #self.b = [np.zeros(self.m) for _ in range(self.n)] # b_1, ..., b_n
        
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
        
        # graph adjacency matrix
        self.adjacency_matrix = np.identity(self.n) * np.diag(self.W) - self.W
        
        # mixing matrix = metropolis weights matrix
        self.mixing_matrix = utils.get_metropolis_weights(self.adjacency_matrix)
        
        # gossip matrix = I - mixing matrix
        self.gossip_matrix = np.identity(self.n) - self.mixing_matrix
        
        # we can choose W to be gossip matrix
        if gossip:
            self.W = self.gossip_matrix
        
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
            s2min_plus = utils.get_s2min_plus(self.bB)
            mu_tildeF = mu_Phi * s2min_plus # mu_tildeF >= mu_Phi * s2min_plus(bB)
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
            s2max = utils.get_s2max(self.bB)
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
        split_indices = np.cumsum(self.dims)[:-1]
        x = np.split(bx, split_indices)
        return x
    
    
    def _get_solution(self):
        """
        Returns:
            x.value: Solution.
            prob.value: Function value at solution.
        """
        raise NotImplementedError
    

class ExampleModel(Model):
    """
    Model from Example 2 in paper
    "Decentralized Proximal Method of Multipliers 
    for Convex Optimization with Coupled Constraints", 2023.
    """
    def __init__(self, 
                 num_nodes: int, 
                 num_cons: int, 
                 dims: List[int], 
                 graph: str = 'ring', 
                 edge_prob: float = None, 
                 nu: float = None,
                 gossip: bool = False) -> None:
        super().__init__(num_nodes, num_cons, dims, graph, edge_prob, nu, gossip)
        
        # set parameters for function
        self.C = [np.random.rand(self.dims[i], self.dims[i]) for i in range(self.n)] # C_1, ..., C_n
        self.d = [np.random.rand(self.dims[i]) for i in range(self.n)] # d_1, ..., d_n
        self.theta = 1e-3 # regularization parameter
        
        self.bC = sp.linalg.block_diag(*self.C) # bC = diag(C_1, ... C_n)
        self.bd = np.hstack(self.d) # bd = col(d_1, ..., d_n)
        
        self._bCT_bC = None # used in grad_F and hess_F
        self._bCT_bd = None # used in grad_F

        self.solution = self._get_solution()

        
    @property
    def bCT_bC(self) -> np.ndarray:
        if self._bCT_bC is None:
            self._bCT_bC = self.bC.T @ self.bC
        return self._bCT_bC

    @property
    def bCT_bd(self) -> np.ndarray:
        if self._bCT_bd is None:
            self._bCT_bd = self.bC.T @ self.bd
        return self._bCT_bd
    
    def F(self, bx):
        """
        Args:
            bx: np.ndarray - Vector of primal variables.
        Returns:
            Function value at point bx.
        """
        a = self.bC @ bx - self.bd
        return 1/2 * (a.T @ a + self.theta * bx.T @ bx)
    
    
    def grad_F(self, bx):
        """
        Args:
            bx: np.ndarray - Vector of primal variables.
        Returns:
            Function gradient at point bx.
        """
        return self.bCT_bC @ bx - self.bCT_bd + self.theta * bx
    
    
    def hess_F(self, bx: np.ndarray = None):
        """
        Args:
            bx: np.ndarray = None - Vector of primal variables.
        Returns:
            Function hessian at point bx.
        """
        return self.bCT_bC + self.theta * np.identity(self.dim)
    
    
    def _get_solution(self):
        """
        Returns:
            xz_star = np.hstack((x.value, z.value)): Solution.
            prob.value: Function value at solution.
        """
        x = cp.Variable(self.dim)
        z = cp.Variable(self.n * self.m)
        
        objective = cp.Minimize(
            1/2 * cp.sum_squares(self.bC @ x - self.bd) 
            + self.theta/2 * cp.sum_squares(x)
        )
        
        constraints = [self.bA_prime @ x + self.bW @ z - self.bb_prime == 0]
        
        prob = cp.Problem(objective, constraints)
        prob.solve()
        
        return np.hstack((x.value, z.value)), prob.value