from time import time
from em_runner import EMRunner
from utils import insert_intercept

import json
import pandas as pd

if __name__ == "__main__":
    # start timer

    start_time = time()
    
    data = pd.read_csv("562606.csv")

    X = data.Price.values
    X = insert_intercept(X.reshape(-1, 1)) # Creates a T by 2 design matrix

    y = data.drop(columns=["Price"]).values # Creates a T by N matrix of purchases

    print(y.shape)

    estimated_params = {}
    max_k = 5

    for k in range(2, max_k + 1):
        print(f"Running EM with K={k} segments")

        em_runner = EMRunner(X=X, y=y, max_iters=500, tol=1e-2, K=k)
        theta, pi, log_likelihood, bic = em_runner.redundant_EM(redundancy=25)
        estimated_params[k] = (theta, pi, log_likelihood, bic)

    # export to json
    
    with open("estimated_params.json", "w") as f:
        json.dump({k: {"theta": theta.tolist(), "pi": pi.tolist(), "log_likelihood": log_likelihood, "BIC": bic} 
                   for k, (theta, pi, log_likelihood, bic) in estimated_params.items()}, f)

    end_time = time()
    print(f"Elapsed time: {end_time - start_time} seconds")