import numpy as np
from matplotlib import pyplot as plt
import os
import pickle
import time


class Timer:
    def __init__(self, name=''):
        self.start_time = 0
        self.end_time = 0
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        print(f"Elapsed time {self.name}: {self.end_time - self.start_time:.2f} seconds")


def save_object(obj, filename):
    # Overwrites any existing file.
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as inp:
        obj = pickle.load(inp)
        return obj
    

def lambda_max(matrix: np.ndarray) -> float:
    """
    Args:
        matrix: np.ndarray - Square matrix.
    Returns:
        lambda_max: float - Maximum eigenvalue of matrix.
    """
    assert matrix.shape[0] == matrix.shape[1], "Matrix must be square"
    eigvals = np.linalg.eigvalsh(matrix)
    lambda_max = eigvals.max()
    return lambda_max


def lambda_min(matrix: np.ndarray) -> float:
    """
    Args:
        matrix: np.ndarray - Square matrix.
    Returns:
        lambda_min: float - Minimum eigenvalue of matrix.
    """
    assert matrix.shape[0] == matrix.shape[1], "Matrix must be square"
    eigvals = np.linalg.eigvalsh(matrix)
    lambda_min = eigvals.min()
    return lambda_min


def lambda_min_plus(matrix: np.ndarray, tol: float = 1e-6) -> float:
    """
    Args:
        matrix: np.ndarray - Square matrix.
        tol: float = 1e-6 - Threshold for consider eigenvalue as 0.
    Returns:
        lambda_min_plus: float - Minimum positive eigenvalue of matrix.
    """
    assert matrix.shape[0] == matrix.shape[1], "Matrix must be square"
    eigvals = np.linalg.eigvalsh(matrix)
    lambda_min_plus = eigvals[eigvals > tol].min()
    return lambda_min_plus


def get_s2min_plus(matrix: np.ndarray, tol: float = 1e-6) -> float:
    """
    Args:
        matrix: np.ndarray - Matrix.
        tol: float = 1e-6 - Threshold for consider singluar value as 0.
    Returns:
        s2min_plus: float - Minimum squared positive singular value of matrix.
    """
    _, sigma, _ = np.linalg.svd(matrix) # singular values of matrix
    sigma = sigma[sigma > tol]
    sigma_squared = sigma ** 2
    s2min_plus = sigma_squared.min()
    return s2min_plus


def get_s2max(matrix: np.ndarray) -> float:
    """
    Args:
        matrix: np.ndarray - Matrix.
    Returns:
        s2max: float - Maximum squared singular value of matrix.
    """
    _, sigma, _ = np.linalg.svd(matrix) # singular values of matrix
    sigma_squared = sigma ** 2
    s2max = sigma_squared.max()
    return s2max


def get_ring_W(num_nodes: int) -> np.ndarray:
    """
    Args:
        num_nodes: int - Number of nodes in the graph.
    Returns:
        W: np.ndarray - Laplacian matrix of the ring graph.
    """
    if num_nodes == 1:
        return np.array([[0]])
    if num_nodes == 2:
        return np.array([[1, -1], [-1, 1]])
    w1 = np.zeros(num_nodes)
    w1[0], w1[-1], w1[1] = 2, -1, -1
    W = np.array([np.roll(w1, i) for i in range(num_nodes)])
    return W


def get_ER_W(num_nodes: int, p: float) -> np.ndarray:
    """
    Args:
        num_nodes: int - Number of nodes in the graph.
        p: float - Probability threshold for Erdos-Renyi graph.
    Returns:
        W: np.ndarray - Laplacian matrix of the connected Erdos-Renyi graph.
    """
    import networkx as nx

    assert p > 0
    while True:
        graph = nx.random_graphs.erdos_renyi_graph(num_nodes, p, directed=False, seed=np.random)
        if nx.is_connected(graph):
            break

    M = nx.to_numpy_array(graph)
    D = np.diag(np.sum(M, axis=1))
    W = D - M  # Laplacian
    
    return W


def get_metropolis_weights(adjacency_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the Metropolis weights for the communication graph.
    W_{ij} = 1 / (1 + max(d_i, d_j)) if (i, j) in E else (1 - sum_{k in N_i} W_{ik}).
    Args:
        adjacency_matrix: np.ndarray - Adjacency matrix of the communication graph.
    Returns:
        metropolis_weights: np.ndarray - Metropolis weights matrix.
    """
    # calculate vertices degrees
    degrees = adjacency_matrix.sum(axis=1)
    # calculate 1 / (1 + max(d_i, d_j))
    tmp = np.array([[1 / (1 + max(degrees[i], degrees[j])) for j in range(len(degrees))] for i in range(len(degrees))])
    # filter (i, j) in E
    tmp = tmp * adjacency_matrix
    # add diagonal elements
    metropolis_weights = tmp + (1 - tmp.sum(axis=1)) * np.identity(tmp.shape[0])
    return metropolis_weights


def plot_logs(output, title):
    """
    Plot 3 graphs: primal variable error, function error, constraints error.
    """
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    ax[0].plot(output['x_err'])
    ax[0].set_yscale('log')
    ax[0].set_xlabel("Iterarion number")
    ax[0].set_ylabel(r"$\| x^k - x^* \|_2^2$")
    ax[0].set_title("Primal variable error")

    ax[1].plot(output['F_err'])
    ax[1].set_yscale('log')
    ax[1].set_xlabel("Iterarion number")
    ax[1].set_ylabel(r"$|F(x^k) - F^*|$")
    ax[1].set_title("Function error")

    ax[2].plot(output['cons_err'])
    ax[2].set_yscale('log')
    ax[2].set_xlabel("Iterarion number")
    ax[2].set_ylabel(r"$\| \sum\limits_{i=1}^{n} (A_i x_i^k - b_i) \|_2$")
    ax[2].set_title("Constraints error")

    plt.suptitle(title, fontsize=24)
    plt.tight_layout()
    plt.show()
    
    
def plot_logs_pd(output, title):
    """
    Plot 4 graphs: primal variable error, function error, constraints error, primal-dual error.
    """
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))

    ax[0][0].plot(output['x_err'])
    ax[0][0].set_yscale('log')
    ax[0][0].set_xlabel("Iterarion number")
    ax[0][0].set_ylabel(r"$\| x^k - x^* \|_2^2$")
    ax[0][0].set_title("Primal variable error")

    ax[0][1].plot(output['F_err'])
    ax[0][1].set_yscale('log')
    ax[0][1].set_xlabel("Iterarion number")
    ax[0][1].set_ylabel(r"$|F(x^k) - F^*|$")
    ax[0][1].set_title("Function error")

    ax[1][0].plot(output['cons_err'])
    ax[1][0].set_yscale('log')
    ax[1][0].set_xlabel("Iterarion number")
    ax[1][0].set_ylabel(r"$\| \sum\limits_{i=1}^{n} (A_i x_i^k - b_i) \|_2$")
    ax[1][0].set_title("Constraints error")

    ax[1][1].plot(output['primal_dual_err'])
    ax[1][1].set_yscale('log')
    ax[1][1].set_xlabel("Iterarion number")
    ax[1][1].set_ylabel(r"$\| \nabla_x G(x^k, y^k) - \mathbf{A}^\top z^k \|_2 $")
    ax[1][1].set_title("Primal-Dual error")

    plt.suptitle(title, fontsize=24)
    plt.tight_layout()
    plt.show()
    
    
def plot_comparison_iteration(results):
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    for name in results.keys():
        ax[0].plot(results[name]['x_err'], label=name)
    ax[0].set_yscale('log')
    ax[0].set_xlabel("Iterarion number")
    ax[0].set_ylabel(r"$\| x^k - x^* \|_2^2$")
    ax[0].set_title("Primal variable error")

    for name in results.keys():
        ax[1].plot(results[name]['F_err'], label=name)
    ax[1].set_yscale('log')
    ax[1].set_xlabel("Iterarion number")
    ax[1].set_ylabel(r"$|F(x^k) - F^*|$")
    ax[1].set_title("Function error")

    for name in results.keys():
        ax[2].plot(results[name]['cons_err'], label=name)
    ax[2].set_yscale('log')
    ax[2].set_xlabel("Iterarion number")
    ax[2].set_ylabel(r"$\| \sum\limits_{i=1}^{n} (A_i x_i^k - b_i) \|_2$")
    ax[2].set_title("Constraints error")
    
    plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")
    plt.suptitle('Comparison', fontsize=24)
    plt.tight_layout()
    plt.show()
    
    
def plot_comparison_time(results):
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    for name in results.keys():
        ax[0].plot(results[name]['ts'], results[name]['x_err'], label=name)
    ax[0].set_yscale('log')
    ax[0].set_xlabel("Time, s")
    ax[0].set_ylabel(r"$\| x^k - x^* \|_2^2$")
    ax[0].set_title("Primal variable error")

    for name in results.keys():
        ax[1].plot(results[name]['ts'], results[name]['F_err'], label=name)
    ax[1].set_yscale('log')
    ax[1].set_xlabel("Time, s")
    ax[1].set_ylabel(r"$|F(x^k) - F^*|$")
    ax[1].set_title("Function error")

    for name in results.keys():
        ax[2].plot(results[name]['ts'], results[name]['cons_err'], label=name)
    ax[2].set_yscale('log')
    ax[2].set_xlabel("Time, s")
    ax[2].set_ylabel(r"$\| \sum\limits_{i=1}^{n} (A_i x_i^k - b_i) \|_2$")
    ax[2].set_title("Constraints error")

    plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")
    plt.suptitle('Comparison', fontsize=24)
    plt.tight_layout()
    plt.show()
    
    
def plot_primal_oracles(output, title):
    
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    ax[0].plot(np.cumsum(output['grad_calls']), output['x_err'])
    ax[0].set_yscale('log')
    ax[0].set_xlabel("Gradient calls")
    ax[0].set_ylabel(r"$\| x^k - x^* \|_2^2$")

    ax[1].plot(np.cumsum(output['mults_A']), output['x_err'])
    ax[1].set_yscale('log')
    ax[1].set_xlabel("Multiplications by $\mathbf{A}$ and $\mathbf{A}^T$")
    ax[1].set_ylabel(r"$\| x^k - x^* \|_2^2$")

    ax[2].plot(np.cumsum(output['communications']), output['x_err'])
    ax[2].set_yscale('log')
    ax[2].set_xlabel("Communications")
    ax[2].set_ylabel(r"$\| x^k - x^* \|_2^2$")

    plt.suptitle(title, fontsize=24)
    plt.tight_layout()
    plt.show()
    
    
def plot_comparison_oracles(results):
    
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    for title in results.keys():
        output = results[title]
        x_vals = np.cumsum(output['grad_calls'])
        y_vals = output['x_err']
        ax[0].plot(x_vals, y_vals, label=title)
    #ax[0].set_xlim(left=0, right=np.cumsum(results['DPMM']['grad_calls'])[-1] * 0.1)
    #ax[0].set_ylim(bottom=results['APAPC']['x_err'][-1], top=results['DPMM']['x_err'][0] * 10)
    ax[0].set_yscale('log')
    ax[0].set_xlabel("Gradient calls")
    ax[0].set_ylabel(r"$\| x^k - x^* \|_2^2$")

    for title in results.keys():
        output = results[title]
        x_vals = np.cumsum(output['mults_A'])
        y_vals = output['x_err']
        ax[1].plot(x_vals, y_vals, label=title)
    #ax[1].set_xlim(left=0, right=np.cumsum(results['DPMM']['mults_A'])[-1] * 0.5)
    #ax[1].set_ylim(bottom=results['APAPC']['x_err'][-1], top=results['DPMM']['x_err'][0] * 10)
    ax[1].set_yscale('log')
    ax[1].set_xlabel("Multiplications by $\mathbf{A}$ and $\mathbf{A}^T$")
    ax[1].set_ylabel(r"$\| x^k - x^* \|_2^2$")

    for title in results.keys():
        output = results[title]
        x_vals = np.cumsum(output['communications'])
        y_vals = output['x_err']
        ax[2].plot(x_vals, y_vals, label=title)
    #ax[2].set_xlim(left=0, right=np.cumsum(results['DPMM']['communications'])[-1])
    #ax[2].set_ylim(bottom=results['APAPC']['x_err'][-1], top=results['DPMM']['x_err'][0] * 10)
    ax[2].set_yscale('log')
    ax[2].set_xlabel("Communications")
    ax[2].set_ylabel(r"$\| x^k - x^* \|_2^2$")
    
    plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")
    plt.suptitle('Comparison', fontsize=24)
    plt.tight_layout()
    plt.show()
    

#########


def generate_matrices_for_condition_number(n, d, condition_number, theta):
    """
    Generate matrices C_i to achieve a specific condition number.

    Parameters:
    - n: Number of blocks.
    - d: Dimension of each block.
    - condition_number: Desired condition number.
    - theta: Regularization parameter.

    Returns:
    - C_list: List of matrices C_i.
    """
    C_list = []

    # Set the eigenvalues to achieve the desired condition number
    lambda_min = 1 + theta
    lambda_max = condition_number * (1 + theta)

    for i in range(n):
        # Generate a random orthogonal matrix U
        U, _ = np.linalg.qr(np.random.randn(d, d))

        # Set the eigenvalues for C_i^T C_i
        if i == 0:
            eigenvalues = np.linspace(lambda_min, lambda_min, d)
        else:
            eigenvalues = np.linspace(lambda_max, lambda_max, d)

        # Construct C_i
        Lambda = np.diag(eigenvalues)
        C_i = U @ np.sqrt(Lambda)

        C_list.append(C_i)

    return C_list