# %%
import numpy as np
from scipy.linalg import block_diag
from numpy.linalg import inv, pinv

class Method:

    def get_network(self, A):
        self.adjacency_matrix = A
        self.n_agents = A.shape[0]

    def get_function(self, R_array, r_array):
        self.R_array = R_array
        self.r_array = r_array
        self.dim_i = R_array[0].shape[0]

    def get_constraints(self, B_array, b_e):
        self.B_array = B_array
        self.b_e = b_e

    def __init__(self, A, R_array, r_array, B_array, b_e):
        self.get_network(A)
        self.get_function(R_array, r_array)
        self.get_constraints(B_array, b_e)

    def f_i(self, i, x_i):
        return x_i @ self.R_array[i] @ x_i + self.r_array[i] @ x_i

    def parse(self, x):
        n_array = np.array([self.dim_i for _ in range(self.n_agents)])
        cum = [0, *np.cumsum(n_array)]
        x_array = []
        for i in range(self.n_agents):
            x_array.append(x[cum[i]:cum[i+1]])
        return np.array(x_array)

    def f(self, x):
        x_array = self.parse(x)
        return np.array([self.f_i(i, x_array[i]) for i in range(self.n_agents)]).sum()
    
    def Proj(self, x):
        y = np.zeros(len(x))
        for i in range(len(x)):
            if x[i] < -1:
                y[i] = -1
            elif -1 <= x[i] <= 1:
                y[i] = x[i]
            elif x[i] > 1:
                y[i] = 1
        return y
    
    def prox(self, x):
        return self.Proj(x)

    def MetropolisWeights(self, E):
        d = E.sum(axis=1)
        W = np.zeros((E.shape[0], E.shape[1]))
        for i in range(E.shape[0]):
            for j in range(E.shape[1]):
                if i == j:
                    continue
                else:
                    if E[i][j] == 1:
                        W[i][j] = 1 / (1 + max(d[i], d[j]))
                    else:
                        W[i][j] = 0
            W[i][i] = 1 - W[i].sum()
        return W

    def get_n_iter(self, n_iter):
        self.n_iter = n_iter

# %%
class Alghunaim(Method):
    
    def __init__(self, A, R_array, r_array, B_array, b_e):
        super().__init__(A, R_array, r_array, B_array, b_e)
        self.K = self.n_agents
        self.E = self.K
        self.Q_k = self.dim_i

    def J(self, k, w_k):
        return w_k @ self.R_array[k] @ w_k + self.r_array[k] @ w_k
    
    def grad_J(self, k, w_k):
        return 2 * self.R_array[k] @ w_k + self.r_array[k]
    
    def grad_J_bar(self, w):
        lst_w_k = self.parse(w)
        return np.hstack(tuple([self.grad_J(k, lst_w_k[k]) for k in range(self.K)]))

    def get_start_point(self, wm1, ym1):
        self.wm1 = wm1
        self.ym1 = ym1

    def get_step_sizes(self, mu_w, mu_y):
        self.mu_w = mu_w
        self.mu_y = mu_y

    def get_B_T_coupled(self):
        A_I = self.adjacency_matrix + np.identity(self.K)
        B_bar_matrix = []

        for e in range(self.E):
            B_bar_array = []
            for k in range(self.K):
                B = []
                for k_bar in np.nonzero(A_I[e])[0]:
                    if k in np.nonzero(A_I[e])[0] and k == k_bar:
                        B.append(self.B_array[e][k].T)
                    else: 
                        B.append(np.zeros((self.Q_k, 1)))
                B_bar_array.append(np.hstack(tuple(B)))
            B_bar_matrix.append(B_bar_array)

        B_bar_matrix_T = [[None for k in range(self.K)] for e in range(self.E)]

        for e in range(self.E):
            for k in range(self.K):
                B_bar_matrix_T[e][k] = B_bar_matrix[k][e]
    
        return np.block(B_bar_matrix_T)

    def get_b_coupled(self):
        A_I = self.adjacency_matrix + np.identity(self.K)
        N = [int(A_I.sum(axis=1)[i]) for i in range(self.K)]
        return np.hstack(tuple([1/N[e]*(np.kron(np.ones(N[e]), self.b_e[e])) for e in range(self.E)]))

    def get_B_T_uncoupled(self):
        B_bar_matrix = []

        for k in range(self.K):
            B = []
            for k_bar in range(self.K):
                if k in range(self.K) and k == k_bar:
                    B.append(self.B_array[k].T)
                else: 
                    B.append(np.zeros((self.Q_k, self.K)))
            B_bar_matrix.append(np.hstack(tuple(B)))

        return np.vstack(B_bar_matrix)

    def get_b_uncoupled(self):
        return 1/self.K*(np.kron(np.ones(self.K), self.b_e))

    def get_B_T(self, coupled=False):
        if coupled == True:
            return self.get_B_T_coupled()
        else:
            return self.get_B_T_uncoupled()
        
    def get_b(self, coupled=False):
        if coupled == True:
            return self.get_b_coupled()
        else:
            return self.get_b_uncoupled()

    def get_A_coupled(self):
        A_I = self.adjacency_matrix + np.identity(self.K)
        N = [int(A_I.sum(axis=1)[i]) for i in range(self.K)]
        A_array = []

        for e in range(self.E):
            lst_e = np.nonzero(A_I[e])[0]
            A_e = self.MetropolisWeights(self.adjacency_matrix[np.ix_(lst_e, lst_e)])
            A_array.append(A_e)

        A_bar_array = []

        for e in range(self.E):
            A_bar_array.append(np.kron(1/2*(np.identity(N[e]) + A_array[e]), np.identity(1)))
            
        return block_diag(*A_bar_array)

    def get_A_uncoupled(self):
        A_e = self.MetropolisWeights(self.adjacency_matrix)
        return np.kron(1/2*(np.identity(self.K) + A_e), np.identity(self.K))
    
    def get_A(self, coupled=False):
        if coupled == True:
            return self.get_A_coupled()
        else:
            return self.get_A_uncoupled()

    def solve(self, coupled=False):
        B_T = self.get_B_T(coupled)
        B = B_T.T
        b = self.get_b(coupled)
        A_bar = self.get_A(coupled)
        A_I = self.adjacency_matrix + np.identity(self.K)
        N = [int(A_I.sum(axis=1)[i]) for i in range(self.K)]

        w0 = self.prox(self.wm1 - self.mu_w * self.grad_J_bar(self.wm1) - self.mu_w * B_T @ self.ym1)
        y0 = self.ym1 + self.mu_y * (B @ w0 - b)
        
        w_i = np.zeros((self.n_iter, self.Q_k * self.K))
        w_i[0] = self.wm1
        w_i[1] = w0

        if coupled == True:
            y_i = np.zeros((self.n_iter, sum(N)))
        else:
            y_i = np.zeros((self.n_iter, self.K * self.E))

        y_i[0] = self.ym1
        y_i[1] = y0
        
        for i in range(2, self.n_iter):
            w_i[i] = self.prox(w_i[i-1] - self.mu_w * self.grad_J_bar(w_i[i-1]) - self.mu_w * B_T @ y_i[i-1])
            y_i[i] = A_bar @ (2 * y_i[i-1] - y_i[i-2] + self.mu_y * B @ (w_i[i] - w_i[i-1]))
        
        return w_i, y_i

# %%
class Huang(Method):

    def __init__(self, A, R_array, r_array, B_array, b_e):
        super().__init__(A, R_array, r_array, B_array, b_e)
        self.N = self.n_agents
        self.m = self.n_agents
        self.n = self.dim_i * self.N
        
        degrees = self.adjacency_matrix.sum(axis=1)
        D = np.diag(degrees)
        self.L = D - self.adjacency_matrix

    def h_i(self, i, x_i):
        return self.B_array[i] @ x_i - 1/self.N * self.b_e
    
    def h(self, x):
        x_array = self.parse(x)
        return np.array([self.h_i(i, x_array[i]) for i in range(self.N)]).sum(axis=1)
    
    def grad_f_i(self, i, x_i):
        return 2 * self.R_array[i] @ x_i + self.r_array[i]

    def grad_h_i(self, i, x_i):
        return self.B_array[i]
    
    def grad_f(self, x):
        x_array = self.parse(x)
        return np.hstack(tuple([self.grad_f_i(i, x_array[i]) for i in range(self.N)]))
    
    def psi_i(self, i, x_i):
        return self.h_i(i, x_i)

    def psi(self, x):
        x_array = self.parse(x)
        return np.array([self.psi_i(i, x_array[i]) for i in range(self.N)]).sum(axis=1)
    
    def grad_psi_i(self, i, x_i):
        return self.grad_h_i(i, x_i)

    def grad_psi(self, x):
        x_array = self.parse(x)
        return np.hstack(tuple([self.grad_psi_i(i, x_array[i]) for i in range(self.N)]))

    def psi_tilde(self, x):
        x_array = self.parse(x)
        return np.hstack(tuple([self.psi_i(i, x_array[i]) for i in range(self.N)]))

    def grad_psi_tilde(self, x):
        x_array = self.parse(x)
        return block_diag(*[self.grad_psi_i(i, x_array[i]) for i in range(self.N)])
    
    def P_Omega_i(self, x_i):
        return self.Proj(x_i)
        
    def P_Omega(self, x):
        x_array = self.parse(x)
        return np.hstack(tuple([self.P_Omega_i(x_array[i]) for i in range(self.N)]))

    def P_Theta_i(self, lmbd_i):
        return lmbd_i

    def P_Theta(self, lmbd):
        return np.hstack(tuple([self.P_Theta_i(lmbd[i:i+self.m]) for i in range(0, self.m * self.N, self.m)]))
    
    def eps(self, k):
        return np.array([10 / (k+1)**2 for _ in range(self.N)])
    
    def get_start_point(self, x0, lmbd0, s0):
        self.x0 = x0
        self.lmbd0 = lmbd0
        self.s0 = s0

    def get_step_sizes(self, k_c):
        self.alpha = 1 / 2 * 1 / (3 * k_c)
        self.beta = 1 / 2 * (1 - 3 * self.alpha * k_c) / (self.alpha * np.linalg.eigvals(self.L).max())

    def solve(self, event_triggered=False):
        xm1 = self.x0
        lmbdm1 = self.lmbd0

        x_k = np.zeros((self.n_iter, self.n))
        lmbd_k = np.zeros((self.n_iter, self.m * self.N))
        s_k = np.zeros((self.n_iter, self.m * self.N))

        x_k[0] = xm1
        x_k[1] = self.x0

        lmbd_k[0] = lmbdm1
        lmbd_k[1] = self.lmbd0

        lmbd_tilde_k = np.zeros((self.n_iter, self.m * self.N))
        lmbd_tilde_k[1] = self.lmbd0

        s_k[1] = self.s0        

        C = np.zeros((self.n_iter, self.N)) # communication numbers
        C[1] = np.ones(self.N)

        k = 1

        while k <= self.n_iter-2:

            # updates
            x_k[k+1] = self.P_Omega(x_k[k] - 2 * self.alpha * (self.grad_f(x_k[k]) + self.grad_psi_tilde(x_k[k]).T @ lmbd_k[k]) + self.alpha * (self.grad_f(x_k[k-1]) + self.grad_psi_tilde(x_k[k-1]).T @ lmbd_k[k-1]))
            lmbd_k[k+1] = self.P_Theta(lmbd_k[k] + 2 * self.alpha * self.psi_tilde(x_k[k]) - self.alpha * self.psi_tilde(x_k[k-1]) - self.alpha * s_k[k] - self.alpha * self.beta * np.kron(self.L, np.identity(self.m)) @ lmbd_tilde_k[k])
            
            if event_triggered == True:
                # test the event-triggered rule
                for i, j in zip(range(0, self.m * self.N, self.m), range(self.N)):
                    if np.linalg.norm(lmbd_tilde_k[k][i:i+self.m] - lmbd_k[k][i:i+self.m]) > self.eps(k)[j]:
                        C[k+1][j] = C[k][j] + 1
                        lmbd_tilde_k[k+1][i:i+self.m] = lmbd_k[k+1][i:i+self.m]
                    else:
                        C[k+1][j] = C[k][j]
                        lmbd_tilde_k[k+1][i:i+self.m] = lmbd_tilde_k[k][i:i+self.m]
            else:
                lmbd_tilde_k[k+1] = lmbd_k[k+1]
                C[k+1] = C[k] + np.ones(self.N)
            
            # update the local update
            s_k[k+1] = s_k[k] + self.beta * np.kron(self.L, np.identity(self.m)) @ lmbd_tilde_k[k+1]
        
            k = k + 1
        
        return x_k, C
    
    def get_communications(self, x_k, C):
        C_s = [int(x) for x in C.mean(axis=1)]
        current = 0
        x_k_unique = []
        for x, C in zip(x_k, C_s):
            if C > current:
                current = C
                x_k_unique.append(x)
            else:
                continue
        C_s_unique = list(set(C_s[1:]))
        return x_k_unique, C_s_unique


# %%
class Carli(Method):

    def __init__(self, A, R_array, r_array, B_array, b_e):
        super().__init__(A, R_array, r_array, B_array, b_e)

        self.N = self.n_agents
        self.H = self.dim_i
        self.M = self.n_agents
        
        R = block_diag(*self.R_array)
        r = np.hstack(tuple(self.r_array))

        self.C = R
        self.q = r

        self.A_array = B_array
        self.A = np.hstack(self.A_array)
        self.b = b_e

        self.A_hat = block_diag(*self.A_array)
        self.b_hat = np.kron(np.ones(self.N), self.b)

    def f(self, x):
        return x @ self.C @ x + self.q @ x

    def newton(self, theta_0: np.ndarray, n_iters: int, F, grad_F, hess_F):
        theta = theta_0
        hessian = hess_F(theta)
        pinv_hessian = inv(hessian)
        for _ in range(n_iters):
            theta = self.Proj(theta - pinv_hessian @ grad_F(theta))
        return theta
        
    def get_start_point(self, x0, l0):
        self.x0 = x0
        self.l0 = l0

    def get_Q_array(self, alpha):
        Q_array = []

        for n in range(self.N):
            Q_n = alpha * (self.N - 1) * (self.A_array[n].T @ self.A_array[n]) + 1 * np.identity(self.H)
            Q_array.append(Q_n)

        return np.array(Q_array)

    def solve(self, alpha, tau):
        self.P = self.MetropolisWeights(self.adjacency_matrix)
        self.P_Ntau = np.kron(np.linalg.matrix_power(self.P, tau), np.identity(self.M))
        I_NM = np.identity(self.N * self.M)
        self.Q_array = self.get_Q_array(alpha)
        self.Q = block_diag(*self.Q_array)

        x_k = np.zeros((self.n_iter, self.N * self.H))
        l_k = np.zeros((self.n_iter, self.N * self.M))

        x_k[0] = self.x0
        l_k[0] = self.l0

        k = 0

        while k <= self.n_iter-2:

            def F(x):
                return self.f(x) + alpha / 2 * np.linalg.norm(self.A_hat @ x + (self.N * self.P_Ntau - I_NM) @ self.A_hat @ x_k[k] - self.b_hat + self.P_Ntau @ l_k[k] / alpha) ** 2 + 1 / 2 * (x - x_k[k]) @ self.Q @ (x - x_k[k])

            def grad_F(x):
                return (2 * self.C + self.Q + alpha * self.A_hat.T @ self.A_hat) @ x + self.q + alpha * self.A_hat.T @ ((self.N * self.P_Ntau - I_NM) @ self.A_hat @ x_k[k] - self.b_hat + self.P_Ntau @ l_k[k] / alpha) - self.Q @ x_k[k]
            
            def hess_F(x):
                return 2 * self.C + self.Q + alpha * self.A_hat.T @ self.A_hat

            x_k[k+1] = self.newton(x_k[k], 1, F, grad_F, hess_F)

            l_k[k+1] = self.P_Ntau @ l_k[k] + alpha * (self.P_Ntau @ self.A_hat @ x_k[k] - self.b_hat)

            k = k + 1

        return x_k, l_k


