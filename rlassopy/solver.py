import numpy as np
import numpy.linalg as la


def lasso_shooting(
    X,
    y,
    lambd,
    psi,
    *,
    sqrt=False,
    intercept=False,
    max_iter=1000,
    opt_tol=1e-10,
    zero_tol=1e-4,
    beta_start=None,
    XX=None,
    Xy=None,
):
    """
    Lasso Shooting algorithm for and sqrt lasso.

    Parameters
    ----------
    XX : ndarray
        Design matrix.
    yy : ndarray
        Response vector.
    lambd : float
        Regularization parameter.
    psi : float
        Penalty loadings.
    sqrt : bool, optional, default False
        If True, use sqrt lasso.
        beta = min ||(y - X @ beta)||_2^2 + lambd ||psi @ beta||_1
    max_iter : int, optional, default: 1000
        Maximum number of iterations.
    opt_tol : float, optional, default: 1e-5
        Optimality tolerance.
    zero_tol : float, optional, default: 1e-4
        Zero tolerance.
    beta_start : ndarray, optional, default: None
        Initial beta estimate.

    Returns
    -------
    beta : ndarray
        Estimated beta.
    """

    # def soft_threshold(x, y):
    #     return np.sign(x) * np.maximum(np.abs(x) - y, 0)

    n, p = X.shape

    if XX is None:
        XX = X.T @ X
    else:
        XX = XX.copy()

    if Xy is None:
        Xy = X.T @ y
    else:
        Xy = Xy.copy()

    if beta_start is None:
        # ridge regression
        if sqrt:
            beta = la.solve(XX * n * 2 + lambd * np.diag(psi**2), Xy * n * 2)
        else:
            beta = la.solve(XX + lambd * np.diag(psi**2), Xy)
        # beta = la.inv(XX) @ Xy
    else:
        beta = beta_start

    if sqrt:
        XX /= n
        Xy /= n
        # calc residuals
        v = y - X @ beta
        mse = np.mean(v**2)

    else:
        XX *= 2
        Xy *= 2

    for _ in range(max_iter):

        beta_old = beta.copy()

        for j in range(p):
            s0 = np.sum(XX[j, :] * beta) - XX[j, j] * beta[j] - Xy[j]

            # TODO: Finish sqrt lasso
            # sqrt lasso
            if sqrt:
                if np.abs(beta[j]) > 0:
                    v += X[:, j] * beta[j]
                    mse = np.mean(v**2)

                if n**2 < (lambd * psi[j]) ** 2 / XX[j, j]:
                    beta[j] = 0

                elif s0 > lambd / (n * psi[j] * np.sqrt(mse)):
                    beta[j] = (
                        (
                            lambd
                            * psi[j]
                            / np.sqrt(n**2 - (lambd * psi[j]) ** 2 / XX[j, j])
                        )
                        * np.sqrt(np.max((mse - (s0**2 / XX[j, j]), 0)))
                        - s0
                    ) / XX[j, j]
                    v -= X[:, j] * beta[j]

                elif s0 < -lambd / (n * psi[j] * np.sqrt(mse)):
                    beta[j] = (
                        -(
                            lambd
                            * psi[j]
                            / np.sqrt(n**2 - (lambd * psi[j]) ** 2 / XX[j, j])
                        )
                        * np.sqrt(np.max((mse - (s0**2 / XX[j, j]), 0)))
                        - s0
                    ) / XX[j, j]
                    v -= X[:, j] * beta[j]

                else:
                    beta[j] = 0

            # lasso
            else:
                # compute the shoot and update beta

                if s0 > (lambd * psi[j]):
                    beta[j] = (lambd * psi[j] - s0) / XX[j, j]

                elif s0 < (-lambd * psi[j]):
                    beta[j] = (-lambd * psi[j] - s0) / XX[j, j]

                else:
                    beta[j] = 0

        # Check for convergence
        if la.norm(beta - beta_old) < opt_tol:
            break

    # set coefficients to zero if below zero tolerance
    beta[abs(beta) < zero_tol] = 0

    return beta
