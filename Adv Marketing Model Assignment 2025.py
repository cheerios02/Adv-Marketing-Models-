# Read the Excel File
import csv
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log1p

with open('562606.csv', 'r') as csvfile:
  # Create a reader object
  csv_reader = csv.reader(csvfile)

def LogL(theta, pi, y, X):
  
    _, N = y.shape
    _, K = theta.shape
    loglik = 0.0

    for i in range(N):
        A = np.zeros(K)
        for s in range(K):
            A[s] = np.sum(y[:, i] * X @ theta[:, s] - np.log1p(np.exp(X @ theta[:, s])))
            
         # apply log-sum-exp trick
        m = np.max(A)
        loglik += m + np.log(np.sum(pi*np.exp(A - m)))   


def Estep(theta, pi, y, X):
    _, N = y.shape
    _, K = theta.shape
    W = np.zeros((N,K))
    m_i = 0.0

    pred = X @ theta
    for i in range(N):
        log_weight = np.zeros(K)
        for s in range(K):
            log_weight[s]= np.log(pi[s]) + np.sum(y[:, i] * pred[:, s] - np.log1p(np.exp(pred[:, s])))

        m_i = np.max(log_weight)
        W[i, :] = np.exp(log_weight - m_i)
        W[i, :] /= np.sum(W[i, :])

    return W
            
def Mstep(y, X, W):
    N, K = W.shape
    theta_est = np.zeros((X.shape(1), K))
    pi_est = np.mean(W, axis=0)

    for k in range(K):
        wk = W[:, k]  # N weights for latent segment k
        
        def objective(theta_vec):
            # logit(X theta)
            eta = X @ theta_vec  # shape: (T,)
            # broadcast weights over time dimension
            # expected log-likelihood contribution
            ll = np.sum([
                wk[i] * (y[:, i] * eta - log1p(np.exp(eta)))
                for i in range(N)
            ])
            return -ll  # scipy minimizes
        
        def gradient(theta_vec):
            eta = X @ theta_vec
            p = expit(eta)  # logistic function
            grad = np.zeros(X.shape(1))
            for i in range(N):
                grad += wk[i] * X.T @ (y[:, i] - p)
            return -grad
        
        # optimize for segment k
        res = minimize(objective,
                       x0=np.zeros(X.shape(1)),
                       jac=gradient,
                       method='BFGS')

        theta_est[:, k] = res.x
        
    return theta_est, pi_est