import numpy as np
from matplotlib import pyplot as plt


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
    sigma_squared = sigma ** 2
    s2min_plus = sigma_squared[sigma_squared > tol].min()
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


def plot_logs(x_err, F_err, cons_err, title):
    """
    Plot 3 graphs: primal variable error, function error, constraints error.
    """
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    ax[0].plot(x_err)
    ax[0].set_yscale('log')
    ax[0].set_xlabel("Iterarion number")
    ax[0].set_ylabel(r"$\| \mathbf{x}^k - \mathbf{x}^* \|_2^2$")
    ax[0].set_title("Primal variable error")

    ax[1].plot(F_err)
    ax[1].set_yscale('log')
    ax[1].set_xlabel("Iterarion number")
    ax[1].set_ylabel(r"$|\tilde{F}(\mathbf{x}^k) - \tilde{F}^*|$")
    ax[1].set_title("Function error")

    ax[2].plot(cons_err)
    ax[2].set_yscale('log')
    ax[2].set_xlabel("Iterarion number")
    ax[2].set_ylabel(r"$\| \mathbf{A}' \mathbf{x}^k + \mathbf{W} \mathbf{z}^k - \mathbf{b}' \|_2$")
    ax[2].set_title("Constraints error")

    plt.suptitle(title, fontsize=24)
    plt.tight_layout()
    plt.show()
    
    
def plot_logs_pd(x_err, F_err, cons_err, primal_dual_err, title):
    """
    Plot 4 graphs: primal variable error, function error, constraints error, primal-dual error.
    """
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))

    ax[0][0].plot(x_err)
    ax[0][0].set_yscale('log')
    ax[0][0].set_xlabel("Iterarion number")
    ax[0][0].set_ylabel(r"$\| \mathbf{x}^k - \mathbf{x}^* \|_2^2$")
    ax[0][0].set_title("Primal variable error")

    ax[0][1].plot(F_err)
    ax[0][1].set_yscale('log')
    ax[0][1].set_xlabel("Iterarion number")
    ax[0][1].set_ylabel(r"$|\tilde{F}(\mathbf{x}^k) - \tilde{F}^*|$")
    ax[0][1].set_title("Function error")

    ax[1][0].plot(cons_err)
    ax[1][0].set_yscale('log')
    ax[1][0].set_xlabel("Iterarion number")
    ax[1][0].set_ylabel(r"$\| \mathbf{A}' \mathbf{x}^k + \mathbf{W} \mathbf{z}^k - \mathbf{b}' \|_2$")
    ax[1][0].set_title("Constraints error")
    
    ax[1][1].plot(primal_dual_err)
    ax[1][1].set_yscale('log')
    ax[1][1].set_xlabel("Iterarion number")
    ax[1][1].set_ylabel(r"$\| K^\top \mathbf{y}^k + \nabla \tilde{F}(\mathbf{x}^k) \|_2$")
    ax[1][1].set_title("Primal-Dual error")

    plt.suptitle(title, fontsize=24)
    plt.tight_layout()
    plt.show()