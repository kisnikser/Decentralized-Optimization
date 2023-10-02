import numpy as np
import utils
import scipy.linalg
import cvxpy as cp

class Model:
    def __init__(self, nodes, dim, cons_rows=2, mu=1, L=1000, lmaxATA=2, graph_model="ring", edge_prob=None):
        self.nodes, self.dim = nodes, dim
        self.mu, self.L = mu, L
        
        self.graph_model = graph_model
        self.edge_prob = edge_prob
        
        if self.graph_model == "ring":
            k = np.arange(0, self.nodes // 2 + 1)
            spectrum = 2 - 2 * np.cos(2 * np.pi * k / self.nodes)
#             print("spectrum:", spectrum)
            self.chi = spectrum.max() / spectrum[1:].min() * 1.01 # for numerical robustness
            print("graph chi:", self.chi)
        elif self.graph_model == "erdos-renyi":
            self.chi = 10 # some random constant
        self.get_bW() # check chi
            
        self.C = []  # np.random.random((nodes, dim, dim))
        for i in range(self.nodes):
            Ci = np.random.random((dim-1, dim)) # mu = 0
            Ci = Ci.T @ Ci
            Ci *= (self.L - self.mu) / utils.lambda_max(Ci)
            Ci += self.mu * np.identity(dim) 
            self.C.append(Ci)
        print("mu=", min([utils.lambda_min(Ci) for Ci in self.C]))
        self.bC = scipy.linalg.block_diag(*self.C)

        # multiply by mu to preserve scaling and numeric stability
        self.d = [np.random.random(self.dim) * self.mu for _ in range(self.nodes)]
        self.bd = np.hstack(self.d)
        
        self.Csum = sum(self.C)
        self.dsum = sum(self.d)
        
        self.cons_rows = cons_rows
        
        lminpATA = 1
        A = utils.get_matrix(cons_rows, dim, np.linspace(lminpATA, lmaxATA, cons_rows))
        self.A = A / lmaxATA ** 0.5 
        self.chitA = lmaxATA / lminpATA
        print("chitA", self.chitA)
            
        s2maxA = 1
        s2minpA = 1 / self.chitA

        # uncomment this to make Ax = b <=> 0 = 0 and make ADOM_affine equivalent to ADOM
        # s2maxA = 0
        # s2minpA = 0 
        # self.A  = np.zeros(self.A.shape)
        # self.chitA = lminpATA = lmaxATA = 0

        self.muH = (1 + s2minpA) / self.L
        self.LH = (1 + s2maxA) / self.mu
        print(f"muH {self.muH}, LH {self.LH}")
        
        In = np.identity(self.nodes)
        self.bA = np.kron(In, self.A)
        self.b = np.random.random(self.cons_rows) 
        self.bb = np.hstack([self.b] * self.nodes)
        
        Id = np.identity(self.dim)
        ones = np.ones((self.nodes, self.nodes))
        self.P = np.kron(In - (ones / self.nodes), Id) # projector on L^\bot
        
        self.f_star, self.x_star = self._get_solution()
        self.f_star_cons, self.x_star_cons = self._get_cons_solution()
        # if A = 0, b = 0
        # self.f_star_cons, self.x_star_cons = self._get_solution()
    
    def split(self, x):
        return x.reshape((self.nodes, self.dim))
    
    def get_bW(self):
        if self.graph_model == "ring":
            W = utils.get_ring_W(self.nodes)  # graph Laplacian. 
        elif self.graph_model == "erdos-renyi":
            W = utils.get_ER_W(self.nodes, self.edge_prob)
            
        perm = np.random.permutation(self.nodes)
        W = W[perm].T[perm].T
        W /= utils.lambda_max(W)
        if not self.chi >= 1 / utils.lambda_min_plus(W):  # check that chi is correctly choosen
            print("eigvals", sorted(np.linalg.eigvals(W)))
            print(f"chi: {self.chi}, actual lmax per lminp: {1 / utils.lambda_min_plus(W)}")
            assert False
        
        # ADOM_PLUS requires scaled Laplacian or (I - M), where M - mixing matrix
        # used scaled laplacian here, TODO: use mixing matrices
        
        Id = np.identity(self.dim)
        self.W = W
        return np.kron(W, Id)
        

    def F(self, bx):
        return (1 / 2 * bx.T @ self.bC @ bx + self.bd @ bx)
    
    def f(self, x):
        return (1 / 2 * x.T @ self.Csum @ x + self.dsum @ x)

    def grad_F(self, bx):
        return (self.bC @ bx + self.bd)

    def _get_solution(self):
        x_star = np.linalg.solve(self.Csum, -self.dsum)
        return self.f(x_star), x_star
    
    def _get_cons_solution(self):
        x = cp.Variable(self.dim)
        f = cp.quad_form(x, self.Csum) / 2 + self.dsum @ x
        cons = [self.A @ x == self.b]
        obj = cp.Minimize(f)
        prob = cp.Problem(obj, cons).solve(verbose=False, eps_abs=1e-10)
        x_star = x.value
        return self.f(x_star), x_star

    def Fstar_grad(self, bu):
        # -(z.T @ t - F(x)) -> min_x
        return np.linalg.solve(
            self.bC, self.bd - bu
        )

def ADOM_PLUS(iters: int, model: Model):
    x_f = x = np.zeros(model.nodes * model.dim)
    y_f = y = np.zeros(model.nodes * model.dim)
    z_f = z = np.zeros(model.nodes * model.dim)
    m = np.zeros(model.nodes * model.dim)

    mu, L = model.mu, model.L

    chi = model.chi

    tau_2 = (mu / L) ** 0.5
    tau_1 = (1 / tau_2 + 0.5) ** -1
    eta = 1 / (L * tau_2)
    alpha = mu / 2
    nu = mu / 2
    beta = 1 / (2 * L)
    sigma_2 = mu ** 0.5 / (16 * chi * L ** 0.5)
    sigma_1 = (1 / sigma_2 + 0.5) ** -1
    gamma = nu / (14 * sigma_2 * chi ** 2)
    delta = 1 / (17 * L)
    theta = nu / (4 * sigma_2)
    zeta = 0.5

    f_err, cons_err, dist = np.zeros(iters), np.zeros(iters), np.zeros(iters)

    for i in range(iters):
        bW = model.get_bW()
        x_g = tau_1 * x + (1 - tau_1) * x_f
        df = model.grad_F(x_g)
        y_g = sigma_1 * y + (1 - sigma_1) * y_f
        z_g = sigma_1 * z + (1 - sigma_1) * z_f

        # solving linear system on x^{k+1} and y^{k+1}
        xi_1 = x + eta * alpha * x_g - eta * (df - nu * x_g)
        xi_2 = y + theta * beta * (df - nu * x_g) - theta * (y_g + z_g) / nu
        xi_3 = xi_2 - theta * xi_1 / (1 + eta * alpha)
        k_1 = theta * eta / (1 + eta * alpha)

        x_prev, y_prev = x, y
        y = xi_3 / (1 + theta * beta + k_1)
        x = (xi_1 + eta * y) / (1 + eta * alpha)

        x_f = x_g + tau_2 * (x - x_prev)
        y_f = y_g + sigma_2 * (y - y_prev)
        z = z + gamma * delta * (z_g - z) - bW @ (gamma * (y_g + z_g) / nu + m)
        m = gamma * (y_g + z_g) / nu + m - bW @ (gamma * (y_g + z_g) / nu + m)  # (I - bW)
        z_f = z_g - zeta * bW @ (y_g + z_g)

        f_err[i] = model.F(x_f) - model.f_star
#         cons_err[i] = np.linalg.norm(bW @ x_f)
        cons_err[i] = np.linalg.norm(model.P @ x_f)
        dist[i] = np.linalg.norm(x_f.reshape(model.nodes, model.dim) - model.x_star, ord=2, axis=1).max()
    return x_f, y_f, z_f, f_err, cons_err, dist

def OPAPC(iters: int, model: Model):
    x_f = x = np.zeros(model.nodes * model.dim)
    y = np.zeros(model.nodes * model.dim)

    mu, L = model.mu, model.L

    chi = model.chi

    T = np.floor(chi)
    c1 = (np.sqrt(chi) - 1) / (np.sqrt(chi) + 1)
    c2 = (chi + 1) / (chi - 1)
    c3 = 2 * chi / ((1 + chi) * utils.lambda_max(W))
    eta = 1 / (4 * tau * L)
    theta = 1 + c1**(2*T) / (eta * (1 + c1**T)**2)
    alpha = mu
    tau = min(1, (1 + c1**T) / (2 * np.sqrt(kappa) * (1 - c1**T)))