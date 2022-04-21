import numpy as np
import numpy.linalg as la


def lasso_shooting(
    X,
    y,
    XX,
    Xy,
    lambd,
    psi,
    starting_values,
    *,
    sqrt=False,
    fit_intercept=True,
    max_iter=1000,
    opt_tol=1e-10,
    zero_tol=1e-4,
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
    opt_tol : float, optional, default: 1e-10
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
    beta = starting_values.copy()

    if not sqrt:

        XX *= 2
        Xy *= 2

        for _ in range(max_iter):
            beta_old = beta.copy()

            for j in range(p):
                S0 = np.sum(XX[j, :] * beta) - XX[j, j] * beta[j] - Xy[j]

                if S0 > (lambd * psi[j]):
                    beta[j] = (lambd * psi[j] - S0) / XX[j, j]

                elif S0 < (-lambd * psi[j]):
                    beta[j] = (-lambd * psi[j] - S0) / XX[j, j]

                else:
                    beta[j] = 0

            if np.sum(np.abs(beta - beta_old)) < opt_tol:
                break

    else:
        XX /= n
        Xy /= n

        # demean X and y
        if fit_intercept:
            X -= np.mean(X, axis=0)
            y -= np.mean(y)

        error = y - X @ beta
        qhat = np.sum(error**2) / n

        for _ in range(max_iter):
            beta_old = beta.copy()

            for j in range(p):
                S0 = np.sum(XX[j, :] * beta) - XX[j, j] * beta[j] - Xy[j]

                # TODO: Finish sqrt lasso
                # sqrt lasso
                if np.abs(beta[j]) > 0:
                    error += X[:, j] * beta[j]
                    qhat = np.mean(error**2)

                if n**2 < (lambd * psi[j]) ** 2 / XX[j, j]:
                    beta[j] = 0

                elif S0 > lambd / (n * psi[j] * np.sqrt(qhat)):
                    beta[j] = (
                        (
                            lambd
                            * psi[j]
                            / np.sqrt(n**2 - (lambd * psi[j]) ** 2 / XX[j, j])
                        )
                        * np.sqrt(np.max((qhat - (S0**2 / XX[j, j]), 0)))
                        - S0
                    ) / XX[j, j]
                    error -= X[:, j] * beta[j]

                elif S0 < -lambd / (n * psi[j] * np.sqrt(qhat)):
                    beta[j] = (
                        -(
                            lambd
                            * psi[j]
                            / np.sqrt(n**2 - (lambd * psi[j]) ** 2 / XX[j, j])
                        )
                        * np.sqrt(np.max((qhat - (S0**2 / XX[j, j]), 0)))
                        - S0
                    ) / XX[j, j]
                    error -= X[:, j] * beta[j]

                else:
                    beta[j] = 0
            # end of loop over j
            error_norm = np.sqrt(np.sum((y - X @ beta) ** 2))
            fobj = error_norm / np.sqrt(n) + (lambd / n) * np.dot(psi, np.abs(beta))

            if error_norm > 1e-10:
                aaa = np.sqrt(n) * (error / error_norm)
                bbb = np.abs(lambd / n * psi - np.abs(X.T @ aaa / n)).T @ np.abs(beta)
                dual = aaa.transpose() @ (y / n) - bbb
            else:
                dual = (lambd / n) * np.dot(psi, np.abs(beta))

            diff = np.sum(np.abs(beta - beta_old))
            if diff < opt_tol and (fobj - dual) < 1e-6:
                break

    # set coefficients to zero if below zero tolerance
    beta[abs(beta) < zero_tol] = 0

    return beta
