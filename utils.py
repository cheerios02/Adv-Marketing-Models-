import numpy as np
import matplotlib.pyplot as plt

def LogL(theta, pi, y, X):
    _, N = y.shape
    K, _ = theta.shape
    loglik = 0.0

    for i in range(N):
        A = np.zeros(K)
        for s in range(K):
            A[s] = np.sum(y[:, i] * (X @ theta.T[:, s]) - np.log1p(np.exp(X @ theta.T[:, s])))
            
         # apply log-sum-exp trick
        m = np.max(A)
        loglik += m + np.log(np.sum(pi*np.exp(A - m)))

    return loglik

def insert_intercept(X: np.ndarray) -> np.ndarray:
    """Insert intercept column (of 1s) to feature matrix X."""
    T = X.shape[0]
    X_new = np.zeros((T, X.shape[1] + 1))
    X_new[:, 0] = 1.0
    X_new[:, 1:] = X

    return X_new

def find_best_loglikelihood(results):
    """Find the best log-likelihood and its corresponding iteration."""
    best_loglik = -np.inf
    best_index = -1

    for i in range(len(results)):
        _, _, loglikelihood, _ = results[i]
        if loglikelihood > best_loglik:
            best_loglik = loglikelihood
            best_index = i

    return best_index

def plot_line_graph(evaluation_metric:list, title:str):
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, 2 + len(evaluation_metric)), evaluation_metric, marker='o')
    plt.xlabel('Number of Segments (K)')
    plt.ylabel(title)
    plt.title(f'{title} vs Number of Segments (K)')
    plt.grid(True)
    plt.show()