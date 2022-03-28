import time

import numpy as np
import solver
import solver_fast


def generate_data(m=80, n=100, sigma=5, density=0.2):
    "Generates data matrix X and observations Y."
    np.random.seed(1)
    beta_star = np.random.randn(n)
    idxs = np.random.choice(range(n), int((1 - density) * n), replace=False)
    for idx in idxs:
        beta_star[idx] = 0
    X = np.random.randn(m, n)
    y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
    return X, y, beta_star


def test_solver():

    X, y, beta_star = generate_data(m=10000, n=500, sigma=5, density=0.2)

    XX = X.T.dot(X)
    Xy = X.T.dot(y)
    psi = np.ones(XX.shape[1])
    starting_vals = np.linalg.inv(XX).dot(Xy)
    lambdas = np.logspace(-5, 5, num=10)
    XX_fast = np.asfortranarray(XX)
    X_fast = np.asfortranarray(X)
    start = time.time()
    for lam in lambdas:
        solver_fast.lasso_shooting(X, y, XX, Xy, lam, psi, starting_vals)
    end = time.time()
    print("Time taken for solver_fast: {}".format(end - start))

    start = time.time()
    for lam in lambdas:
        solver_fast.lasso_shooting(X, y, XX, Xy, lam, psi, starting_vals, sqrt=True)
    end = time.time()
    print("Time taken for solver_fast sqrt: {}".format(end - start))

    start = time.time()
    for lam in lambdas:
        solver.lasso_shooting(X, y, XX, Xy, lam, psi, starting_vals)
    end = time.time()
    print("Time taken for solver: {}".format(end - start))

    start = time.time()
    for lam in lambdas:
        solver.lasso_shooting(X, y, XX, Xy, lam, psi, starting_vals, sqrt=True)
    end = time.time()
    print("Time taken for solver sqrt: {}".format(end - start))


if __name__ == "__main__":
    test_solver()
