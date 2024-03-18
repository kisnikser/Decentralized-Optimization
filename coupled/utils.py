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
    lambda_max = eigvals[-1]
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
    lambda_min = eigvals[0]
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


def plot_logs(x_err, F_err, cons_err, title):
    """
    Plot 3 graphs: primal variable error, function error, constraints error.
    """
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    ax[0].plot(x_err)
    ax[0].set_yscale('log')
    ax[0].set_xlabel("Iterarion number")
    ax[0].set_ylabel(r"$\| \mathbf{x}^k - \mathbf{x}^* \|_2$")
    ax[0].set_title("Primal variable error")

    ax[1].plot(F_err)
    ax[1].set_yscale('log')
    ax[1].set_xlabel("Iterarion number")
    ax[1].set_ylabel(r"$\tilde{F}(\mathbf{x}^k) - \tilde{F}^*$")
    ax[1].set_title("Function error")

    ax[2].plot(cons_err)
    ax[2].set_yscale('log')
    ax[2].set_xlabel("Iterarion number")
    ax[2].set_ylabel(r"$\| \mathbf{A}' \mathbf{x}^k + \mathbf{W} \mathbf{z}^k - \mathbf{b}' \|_2$")
    ax[2].set_title("Constraints error")

    plt.suptitle(title, fontsize=24)
    plt.tight_layout()
    plt.show()