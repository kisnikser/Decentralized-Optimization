import numpy as np
import cvxpy as cp
import scipy as sp
import scipy.stats as st
from typing import List, Optional
import utils

# for dataset downloading
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
    
    
###################################################################################################
    
    
class Model:
    """
    In this class we use notation from the our paper.
    """
    def __init__(self, 
                 num_nodes: int,
                 num_cons: int,
                 d: int,
                 graph: str = 'ring',
                 edge_prob: float = None,
                 gossip: bool = False) -> None:
        """
        Args:
            num_nodes: int - Number of nodes in the graph (n).
            num_cons: int - Number of constraints rows (m).
            d: int - Dimension of the local variables (d).
            graph: str = 'ring' - Network graph model.
            edge_prob: float = None - Probability threshold for Erdos-Renyi graph.
            gossip: bool = False - W will be gossip matrix defined by metropolis weights.
        """
        
        self.n = num_nodes # n
        self.m = num_cons # m
        self.d = d # d
        self.dim = self.n * self.d # d + ... + d = nd
    
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
        self.W_times_I = np.kron(self.W, Im) # bW = W x Im
        
        self._mu_f = None # strongly convexity constant for the function
        self._L_f = None # gradient Lipschitz constant for the function
        self._kappa_f = None # condition number of the function
        
        self._mu_A = None # strongly convexity constant A
        self._L_A = None # gradient Lipschitz constant A
        self._kappa_A = None # condition number A
        
        self._mu_W = None # strongly convexity constant W
        self._L_W = None # gradient Lipschitz constant W
        self._kappa_W = None # condition number W
    
        self._r = None # augmentation parameter
  
    ### PARAMETERS OF FUNCTION
  
    @property
    def mu_f(self) -> float:
        if self._mu_f is None:
            self._mu_f = utils.lambda_min(self.hess_F())
        return self._mu_f
  
    @property
    def L_f(self) -> float:
        if self._L_f is None:
            self._L_f = utils.lambda_max(self.hess_F())
        return self._L_f
    
    @property
    def kappa_f(self) -> float:
        if self._kappa_f is None:
            self._kappa_f = self.L_f / self.mu_f
        return self._kappa_f
    
    ### PARAMETERS OF A
    
    @property
    def mu_A(self) -> float:
        if self._mu_A is None:
            bS = sum([Ai @ Ai.T for Ai in self.A]) / self.n
            self._mu_A = utils.lambda_min_plus(bS)
        return self._mu_A
  
    @property
    def L_A(self) -> float:
        if self._L_A is None:
            A_norms = [utils.get_s2max(Ai) for Ai in self.A]
            self._L_A = max(A_norms)
        return self._L_A
    
    @property
    def kappa_A(self) -> float:
        if self._kappa_A is None:
            self._kappa_A = self.L_A / self.mu_A
        return self._kappa_A
    
    ### PARAMETERS OF W
    
    @property
    def mu_W(self) -> float:
        if self._mu_W is None:
            self._mu_W = utils.lambda_min_plus(self.W)
        return self._mu_W
  
    @property
    def L_W(self) -> float:
        if self._L_W is None:
            self._L_W = utils.lambda_max(self.W)
        return self._L_W
    
    @property
    def kappa_W(self) -> float:
        if self._kappa_W is None:
            self._kappa_W = self.L_W / self.mu_W
        return self._kappa_W
    
    ### AUGMENTATION PARAMETER (see Lemma 1)
    
    @property
    def r(self) -> float:
        if self._r is None:
            self._r = self.mu_f / (2 * self.L_A)
        return self._r
    
    ### ORACLE
    
    def F(self, x):
        """
        Args:
            x: np.ndarray - Vector of primal variables.
        Returns:
            Function value at point x.
        """
        raise NotImplementedError
    
    
    def G(self, x, y):
        """
        Args:
            x: np.ndarray - Vector of primal variables.
            y: np.ndarray - Vector of additional variables.
        Returns:
            Augmented function value at point (x, y).
        """
        return (
            self.F(x)
            + self.r / 2 * np.linalg.norm(self.bA @ x + y - self.bb)**2
        )
        
        
    def grad_F(self, x):
        """
        Args:
            x: np.ndarray - Vector of primal variables.
        Returns:
            Function gradient at point x.
        """
        raise NotImplementedError

        
    def grad_G_x(self, x, y):
        """
        Args:
            x: np.ndarray - Vector of primal variables.
            y: np.ndarray - Vector of additional variables.
        Returns:
            Augmented function gradient wrt x at point (x, y).
        """
        return self.grad_F(x) + self.r * self.bA.T @ (self.bA @ x + y - self.bb)
    
    
    def grad_G_y(self, x, y):
        """
        Args:
            x: np.ndarray - Vector of primal variables.
            y: np.ndarray - Vector of additional variables.
        Returns:
            Augmented function gradient wrt y at point (x, y).
        """
        return self.r * (self.bA @ x + y - self.bb)
    
    
    def hess_F(self, x: Optional[np.ndarray] = None):
        """
        Args:
            x: Optional[np.ndarray] = None - Vector of primal variables.
        Returns:
            Function hessian at point x.
        """
        raise NotImplementedError

    
    def _get_solution(self):
        """
        Returns:
            x_star: Solution.
            F_star: Function value at solution.
        """
        raise NotImplementedError
    

###################################################################################################    

    
class ExampleModel(Model):
    """
    Model from Example 2 in paper
    "Decentralized Proximal Method of Multipliers 
    for Convex Optimization with Coupled Constraints", 2023.
    """
    def __init__(self, 
                 num_nodes: int,
                 num_cons: int,
                 d: int,
                 graph: str = 'ring',
                 edge_prob: Optional[float] = None,
                 gossip: bool = False) -> None:
        """
        Args:
            num_nodes: int - Number of nodes in the graph (n).
            num_cons: int - Number of constraints rows (m).
            d: int - Dimension of the local variables (d).
            graph: str = 'ring' - Network graph model.
            edge_prob: Optional[float] = None - Probability threshold for Erdos-Renyi graph.
            gossip: bool = False - W will be gossip matrix defined by metropolis weights.
        """
        super().__init__(num_nodes, num_cons, d, graph, edge_prob, gossip)   
        
        self.dimensions = [self.d for _ in range(self.n)]
        
        # set parameters for function
        self.C = [np.random.rand(self.d, self.d) for _ in range(self.n)] # C_1, ..., C_n
        self.d_ = [np.random.rand(self.d) for _ in range(self.n)] # d_1, ..., d_n
        self.theta = 1e-3 # regularization parameter
        
        self.bC = sp.linalg.block_diag(*self.C) # bC = diag(C_1, ... C_n)
        self.bd = np.hstack(self.d_) # bd = col(d_1, ..., d_n)
        
        self._bCT_bC = None # used in grad_F and hess_F
        self._bCT_bd = None # used in grad_F
        
        # set parameters for constraints
        self.A = [np.random.randn(self.m, self.d) for _ in range(self.n)] # A_1, ..., A_n
        self.b = [np.random.randn(self.m) for _ in range(self.n)] # b_1, ..., b_n

        self.bA = sp.linalg.block_diag(*self.A) # bA = diag(A_1, ..., A_n)
        self.bb = np.hstack(self.b) # bb = col(b_1, ..., b_n)       

        self.A_hstacked = np.hstack(self.A)
        self.b_sum = np.sum(self.b, axis=0)

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
    
    
    def F(self, x):
        """
        Args:
            x: np.ndarray - Vector of primal variables.
        Returns:
            Function value at point x.
        """
        a = self.bC @ x - self.bd
        return 1/2 * (a.T @ a + self.theta * x.T @ x)
    
    
    def grad_F(self, x):
        """
        Args:
            x: np.ndarray - Vector of primal variables.
        Returns:
            Function gradient at point x.
        """
        return self.bCT_bC @ x - self.bCT_bd + self.theta * x
    
    
    def hess_F(self, x: Optional[np.ndarray] = None):
        """
        Args:
            x: Optional[np.ndarray] = None - Vector of primal variables.
        Returns:
            Function hessian at point x.
        """
        return self.bCT_bC + self.theta * np.identity(self.dim)
    
    
    def _get_solution(self):
        """
        Returns:
            x_star: Solution.
            F_star: Function value at solution.
        """
        x = cp.Variable(self.dim)
        
        objective = cp.Minimize(
            1/2 * cp.sum_squares(self.bC @ x - self.bd) 
            + self.theta/2 * cp.sum_squares(x)
        )
        
        constraints = [self.A_hstacked @ x - self.b_sum == 0]
        
        prob = cp.Problem(objective, constraints)
        prob.solve()
        
        x_star = x.value
        F_star = prob.value
        
        return x_star, F_star
    

###################################################################################################


class VFL(Model):
    """
    Model for Vertical Federative Learning Problem.
    """
    def __init__(self,
                 num_nodes: int,
                 title: str = 'mushrooms',
                 train_size: float = 0.7,
                 graph: str = 'ring',
                 edge_prob: Optional[float] = None,
                 gossip: bool = False,
                 labels_distribution: bool = False) -> None:
        """
        Args:
            num_nodes: int - Number of nodes in the graph (n).
            title: str = 'mushrooms' - Title of the dataset
            train_size: float = 0.7 - Train sample part
            graph: str = 'ring' - Network graph model.
            edge_prob: Optional[float] = None - Probability threshold for Erdos-Renyi graph.
            gossip: bool = False - W will be gossip matrix defined by metropolis weights.
            labels_distribution: bool = False - Labels will be distributed between devices.
        """
        
        feature_matrix, labels = self._get_dataset(title, train_size)
        
        num_features = feature_matrix.shape[1]
        num_cons = feature_matrix.shape[0]
        
        assert num_features % num_nodes == 0, "Number of features must be divided by number of devices"
        
        d = num_features // num_nodes      
        
        super().__init__(num_nodes, num_cons, d, graph, edge_prob, gossip)   
        
        self.dim = self.n * self.d + self.m     
        
        self.Features = np.split(feature_matrix, self.n, axis=1) # [bF_1,  ..., bF_n]
        self.l = labels # l
        self.lmbd = 1e-3 # lambda - regularization parameter
        
        self.bF = np.hstack(self.Features)
        
        self.dimensions = [self.d for _ in range(self.n)]
        self.dimensions[0] = self.d + self.m
        self.labels_distribution = labels_distribution
        if self.labels_distribution:
            self.num_samples = self.split_number(self.m, self.n)
            self.dimensions = [self.d + self.num_samples[i] for i in range(self.n)]
        
        self._get_constraints() # -> self.A, self.b, self.bA, self.bb
        
        self.solution = self._get_solution()
        
    @staticmethod
    def split_number(m, n):
        """
        Divide m between n devices.
        """
        # Calculate the quotient and remainder of m divided by n
        quotient = m // n
        remainder = m % n
        # Create a list of n elements, each initially set to the quotient
        split = [quotient] * n
        # Add 1 to the first 'remainder' elements of the list
        for i in range(remainder):
            split[i] += 1
        return split
    
    
    def _get_dataset(self, 
                     title: str = 'mushrooms', 
                     train_size: float = 0.7):
        """
        Download the dataset `title`.
        Then split it into train and test.
        After that returns feature matrix and labels from the train sample.
        
        Args:
            title: str = 'mushrooms' - Title of the dataset
            train_size: float = 0.7 - Train sample part
        Returns:
            feature_matrix: np.ndarray - Matrix objects-features from train part of the dataset
            labels: np.ndarray - Vector of labels from train part of the dataset
        """
        
        if title == 'mushrooms':
            dataset = '../data/mushrooms.txt'
            data = load_svmlight_file(dataset)
            X, y = data[0].toarray(), data[1]
            y = 2 * y - 3 # -1 and 1
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=train_size, random_state=42
            )
            
        elif title == 'synthetic':
            # synthetic linear regression dataset
            
            n_features = 14 * 2
            n_samples = 14 * 10
            
            mu_x = np.zeros(n_features)
            Sigma_x = np.identity(n_features)
            
            alpha = 1
            sigma2 = 1

            X = st.multivariate_normal(mean=mu_x, cov=Sigma_x).rvs(size=n_samples)
            w = st.multivariate_normal(mean=np.zeros(n_features), cov=alpha**(-1)*np.identity(n_features)).rvs(size=1)
            eps = st.multivariate_normal(mean=np.zeros(n_samples), cov=sigma2*np.identity(n_samples)).rvs(size=1)
            y = X @ w + eps
            
            X_train, y_train = X, y
            
        elif title == 'two':
            
            n_features = 2 * 5
            n_samples = 2 * 10
            
            mu_x = np.zeros(n_features)
            Sigma_x = np.identity(n_features)
            
            alpha = 1
            sigma2 = 1

            X = st.multivariate_normal(mean=mu_x, cov=Sigma_x).rvs(size=n_samples)
            w = st.multivariate_normal(mean=np.zeros(n_features), cov=alpha**(-1)*np.identity(n_features)).rvs(size=1)
            eps = st.multivariate_normal(mean=np.zeros(n_samples), cov=sigma2*np.identity(n_samples)).rvs(size=1)
            y = X @ w + eps
            
            X_train, y_train = X, y
            
        else:
            raise NotImplementedError
       
        feature_matrix = X_train
        labels = y_train
     
        return feature_matrix, labels
    
    
    def _get_constraints(self):
        """
        Create constraints for the VFL problem.
        """

        self.A = [] # list for A_1, ..., A_n
        
        if not self.labels_distribution:
        
            A_1 = np.hstack((self.Features[0], -np.identity(self.m)))
            self.A.append(A_1)
            
            for i in range(1, self.n):
                self.A.append(self.Features[i])
        
        else:
            
            for i in range(self.n):
                dim_z_i = self.num_samples[i]
                left = self.Features[i]
                zeros_dim_i = np.zeros((dim_z_i, dim_z_i))
                ident_dim_i = np.identity(dim_z_i)
                right_matrices = [zeros_dim_i for _ in range(self.n)]
                right_matrices[i] = -ident_dim_i
                right = np.vstack(right_matrices)
                A_i = np.hstack((left, right))
                self.A.append(A_i)
        
        self.b = [np.zeros(self.m) for _ in range(self.n)] # b_1, ..., b_n
        #######
        self.bA = sp.linalg.block_diag(*self.A) # bA = diag(A_1, ..., A_n)
        self.bb = np.hstack(self.b) # bb = col(b_1, ..., b_n)       
        #######
        self.A_hstacked = np.hstack(self.A)
        self.b_sum = np.sum(self.b, axis=0)
    
    
    def _split_vector(self, x):
        """
        In VFL we have the vector x as stacked w and z.
        This stack can be different.
        So we should understand what indices correspond to w and z.
        
        Args:
            x: np.ndarray - Vector of primal variables.
        Returns:
            z: np.ndarray - Vector z (see notation).
            w: np.ndarray - Weights vector.
        """
        
        if not self.labels_distribution:
            # in this case vector x = col(x_1, ..., x_n)
            # x_1 = col(w_1, z)
            # x_i = w_i,  i = 2, ..., n
            x_1 = x[:self.d+self.m]
            w_1 = x_1[:self.d]
            w = np.hstack((w_1, x[self.d+self.m:]))
            z = x_1[self.d:]
        else:
            # in this case vector x = col(x_1, ..., x_n)
            # x_i = col(w_i, z_i),  i = 1, ..., n
            
            xs = np.split(x, np.cumsum(self.dimensions)[:-1])

            z = []
            w = []
                        
            for x in xs:
                w.append(x[:self.d])
                z.append(x[self.d:])
            
            z = np.hstack(z)
            w = np.hstack(w)
            
        return (z, w)
    
    
    def _rearrange_vector(self, g):
        """
        It is the inversed function to _split_vector(x).
        It is easier to calculate F(), grad_F() and hess_F() if form of x = col(z, w).
        So after calculating in such way we have to rearrange it to the initial form.
        
        Args:
            g: np.ndarray - Vector in the form of col(z, w).
        Returns:
            g_initial: np.ndarray - Vector in the initial form of primal variables.
        """
        
        g_z = g[:self.m]
        g_w = g[self.m:]
    
        if not self.labels_distribution:
            # in this case vector x = col(x_1, ..., x_n)
            # x_1 = col(w_1, z)
            # x_i = w_i,  i = 2, ..., n
            
            g_w_1 = g_w[:self.d]
            g_x_1 = np.hstack((g_w_1, g_z))
            g_initial = np.hstack((g_x_1, g_w[self.d:]))
            
        else:
            # in this case vector x = col(x_1, ..., x_n)
            # x_i = col(w_i, z_i),  i = 1, ..., n
    
            g_initial = []
            g_zs = np.split(g_z, np.cumsum(self.num_samples)[:-1])
            g_ws = np.split(g_w, self.n)
                        
            for (g_z_i, g_w_i) in zip(g_zs, g_ws):
                g_x_i = np.hstack((g_w_i, g_z_i))
                g_initial.append(g_x_i)
            
            g_initial = np.hstack(g_initial)
    
        return g_initial
    
    def _rearrange_matrix(self, H):
        """
        It is easier to calculate F(), grad_F() and hess_F() if form of x = col(z, w).
        So after calculating in such way we have to rearrange it to the initial form.
        
        Actually, this function is only for hessian of the function, that is in the form
        (where x = col(z, w)) as hess_F = diag(I_m, 2*lambda*I_nd)
        
        Args:
            H: np.ndarray - Block matrix in the form of col(z, w).
        Returns:
            H_initial: np.ndarray - Block matrix in the initial form of primal variables.
        """
        
        H_z = H[:self.m, :self.m]
        H_w = H[self.m:, self.m:]
    
        if not self.labels_distribution:
            # in this case vector x = col(x_1, ..., x_n)
            # x_1 = col(w_1, z)
            # x_i = w_i,  i = 2, ..., n
            H_w_1 = H_w[:self.d, :self.d]
            H_x_1 = sp.linalg.block_diag(H_w_1, H_z)
            H_initial = sp.linalg.block_diag(H_x_1, H_w[self.d:, self.d:])
        else:
            # in this case vector x = col(x_1, ..., x_n)
            # x_i = col(w_i, z_i),  i = 1, ..., n
            cumsum = np.cumsum([0, *self.num_samples])
            H_initial = []
                        
            for i in range(self.n):
                left = cumsum[i]
                right = cumsum[i+1]
                H_z_i = H_z[left:right, left:right]
                H_w_i = H_w[i*self.d:i*self.d+self.d, i*self.d:i*self.d+self.d]
                H_x_i = sp.linalg.block_diag(H_w_i, H_z_i)
                H_initial.append(H_x_i)
            
            H_initial = sp.linalg.block_diag(*H_initial)

    
        return H_initial
    
    def F(self, x):
        """
        Args:
            x: np.ndarray - Vector of primal variables.
        Returns:
            Function value at point x.
        """
        z, w = self._split_vector(x)
        return 1 / 2 * np.linalg.norm(z - self.l)**2 + self.lmbd * np.linalg.norm(w)**2
        
    
    def grad_F(self, x):
        """
        Args:
            x: np.ndarray - Vector of primal variables.
        Returns:
            Function gradient at point x.
        """
        
        z, w = self._split_vector(x)
        g_z = z - self.l
        g_w = 2 * self.lmbd * w
        g = np.hstack((g_z, g_w))
        h = self._rearrange_vector(g)
        
        return h
    
    
    def hess_F(self, x: Optional[np.ndarray] = None):
        """
        Args:
            x: Optional[np.ndarray] = None - Vector of primal variables.
        Returns:
            Function hessian at point x.
        """
        
        I_m = np.identity(self.m)
        I_nd = np.identity(self.n * self.d)
        
        H = sp.linalg.block_diag(I_m, 2 * self.lmbd * I_nd)
        H_initial = self._rearrange_matrix(H)
        
        return H_initial
    
    
    def _get_solution(self):
        """
        Returns:
            x_star: Solution.
            F_star: Function value at solution.
        """
        w = cp.Variable(self.n * self.d)
        z = cp.Variable(self.m)
        
        objective = cp.Minimize(
            1/2 * cp.sum_squares(z - self.l) 
            + self.lmbd * cp.sum_squares(w)
        )
        
        constraints = [
            self.bF @ w == z
        ]
        
        prob = cp.Problem(objective, constraints)
        prob.solve()
        F_star = prob.value
        
        x_star = self._rearrange_vector(np.hstack((z.value, w.value)))
        
        return x_star, F_star
    
    
###################################################################################################