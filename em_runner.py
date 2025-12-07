import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log1p
from utils import LogL, find_best_loglikelihood

class EMRunner:
    """
    A class to run the Expectation-Maximization (EM) algorithm.
    """
    def __init__(self, X, y, max_iters=1000, tol=1e-4, K=10):
        self.X = X # design matrix
        self.y = y # purchase matrix
        self.K = K # segments
        self.max_iters = max_iters
        self.tol = tol
        self.log_likelihoods = []

        # Initialize model parameters
        self.initialize_parameters()

    def initialize_parameters(self):
        """
        This function initializes the model parameters: 
        - Theta from a normal distribution with variance of 0.1
        - Pi as equal mixing proportions
        """
        # theta shape: (K, 2)
        theta = np.zeros((self.K, 2))

        for s in range(self.K):
            alpha = np.random.normal(0, 0.1, 1)  # intercept
            beta = np.random.normal(0, 0.1, 1)   # slope
            theta[s, 0] = alpha
            theta[s, 1] = beta

        pi = np.ones(self.K) / self.K  # equal mixing proportions

        return theta, pi

    def Estep(self, theta, pi, y:np.ndarray, X:np.ndarray):
        _, N = y.shape
        K, _ = theta.shape

        W = np.zeros((N,K))
        m_i = 0.0

        pred = X @ theta.T
        for i in range(N):
            log_weight = np.zeros(K)
            for s in range(K):
                log_weight[s]= np.log(pi[s]) + np.sum(y[:, i] * pred[:, s] - np.log1p(np.exp(pred[:, s])))

            m_i = np.max(log_weight)
            W[i, :] = np.exp(log_weight - m_i)
            W[i, :] /= np.sum(W[i, :])

        return W
            
    def Mstep(self, y, X, W):
        N, K = W.shape
        theta_est = np.zeros((K, X.shape[1]))
        pi_est = np.mean(W, axis=0)

        for k in range(K):
            wk = W[:, k]  # N weights for latent segment k
            
            def objective(theta_vec):
                # logit(X theta)
                eta = X @ theta_vec.T  
                # broadcast weights over time dimension
                # expected log-likelihood contribution
                ll = np.sum([
                    wk[i] * (y[:, i] * eta - log1p(np.exp(eta)))
                    for i in range(N)
                ])
                return -ll
            
            def gradient(theta_vec):
                eta = X @ theta_vec.T
                p = expit(eta)  # logistic function
                grad = np.zeros(X.shape[1])
                for i in range(N):
                    grad += wk[i] * X.T @ (y[:, i] - p)
                return -grad
            
            # optimize for segment k
            res = minimize(objective,
                        x0=np.zeros(X.shape[1]),
                        jac=gradient,
                        method='BFGS')

            theta_est[k, :] = res.x
            
        return theta_est, pi_est
        
    def EM(self):
        current_theta, current_pi = self.initialize_parameters()

        for iteration in range(self.max_iters):
            # run E-step
            W = self.Estep(current_theta, current_pi, self.y, self.X)

            # run M-step
            new_theta, new_pi = self.Mstep(self.y, self.X, W)
            current_theta, current_pi = new_theta, new_pi

            # compute log-likelihood
            log_likelihood = LogL(new_theta, new_pi, self.y, self.X)
            self.log_likelihoods.append(log_likelihood)

            print(f"Iteration {iteration+1}, Log-Likelihood: {log_likelihood}")

            # check for convergence
            if iteration > 0 and abs(self.log_likelihoods[-1] - self.log_likelihoods[-2]) < self.tol:
                return current_theta, current_pi, log_likelihood

        print("EM did not converge within the maximum number of iterations.")
            
        return current_theta, current_pi, log_likelihood  # did not converge within max_iters

    def redundant_EM(self, redundancy=25):

        results = np.zeros(redundancy, dtype=object)

        for i in range(redundancy):
            print(f"EM Attempt {i+1} of {redundancy} with k={self.K}")

            theta, pi, log_likelihood = self.EM()
            bic = -2 * log_likelihood + self.K * np.log(self.y.shape[1] * self.y.shape[0])

            results[i] = (theta, pi, log_likelihood, bic)

        # find the entry with the highest log-likelihood
        best_index = find_best_loglikelihood(results)

        return results[best_index]