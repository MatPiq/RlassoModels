import numpy as np
import numpy.linalg as la


def lasso_shooting(
    X,
    y,
    lambd,
    psi,
    *,
    sqrt=False,
    max_iter=1000,
    opt_tol=1e-5,
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
    n, p = X.shape

    if XX is None:
        XX = X.T @ X

    if Xy is None:
        Xy = X.T @ y

    if beta_start is None:
        beta = la.inv(XX) @ Xy
    else:
        beta = beta_start

    if sqrt:
        XX /= n
        Xy /= n

    else:
        XX *= 2
        Xy *= 2

    for _ in range(max_iter):
        beta_old = beta
        for j in range(p):
            s0 = np.sum(XX[j, :] * beta) - XX[j, j] * beta[j] - Xy[j]

            # TODO: Finish sqrt lasso
            # sqrt lasso
            if sqrt:
                if n**2 < (lambd * psi[j]) ** 2 / XX[j, j]:
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
