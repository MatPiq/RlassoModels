import logging

import cvxpy as cp
import numpy as np
import scipy.stats as st
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Rlasso(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        *,
        post=True,
        sqrt=False,
        fit_intercept=True,
        cov_type="nonrobust",
        x_dependent=False,
        lasso_psi=False,
        n_corr=5,
        max_iter=2,
        n_sim=5000,
        c=1.1,
        gamma=None,
        zero_tol=1e-4,
        convergence_tol=1e-4,
        verbose=False,
        solver_opts=None,
    ):
        self.post = post
        self.sqrt = sqrt
        self.fit_intercept = fit_intercept
        self.cov_type = cov_type
        self.x_dependent = x_dependent
        self.lasso_psi = lasso_psi
        self.n_corr = n_corr
        self.max_iter = max_iter
        self.n_sim = n_sim
        self.c = c
        self.gamma = gamma
        self.zero_tol = zero_tol
        self.convergence_tol = convergence_tol
        self.verbose = verbose
        self.solver_opts = solver_opts or {}

    def _psi_calc(self, X, n, v=None):

        # TODO Implement cluster robust covariance
        # loadings for sqrt lasso
        if self.sqrt:

            if self.cov_type == "nonrobust":
                psi = np.sqrt(np.mean(X**2, axis=0))

            # heteroscedastic robust case
            elif self.cov_type == "robust" and v is not None:
                Xv2 = np.einsum("ij, i -> j", X**2, v**2)
                psi_1 = np.sqrt(Xv2 / n)
                psi_2 = np.sqrt(Xv2 / np.sum(v**2))
                psi = np.maximum(psi_1, psi_2)
            # clustered
            else:
                raise NotImplementedError(
                    "Cluster robust loadings not \
                                            implemented"
                )

        else:
            if self.cov_type == "nonrobust":
                psi = np.sqrt(np.mean(X**2, axis=0))
            elif self.cov_type == "robust" and v is not None:

                Xe2 = np.einsum("ij, i -> j", X**2, v**2)
                psi = np.sqrt(Xe2 / n)

            else:
                raise NotImplementedError(
                    "Cluster robust loadings not \
                                            implemented"
                )

        return np.diag(psi)

    def _lambd_calc(
        self,
        n,
        p,
        *,
        v=None,
        s1=None,
        X=None,
        psi=None,
    ):

        # catch parameters are provided

        if self.cov_type == "nonrobust" and s1 is None:
            raise ValueError(f"RMSE must be provided for {self.cov_type}")

        if self.x_dependent and psi is None:
            raise ValueError("X must be provided for x_dependent")

        if self.sqrt:
            lasso_factor = self.c
            pass

        else:

            lasso_factor = 2 * self.c
            # homoscedasticity and x-independent case
            if self.cov_type == "nonrobust" and not self.x_dependent:
                assert s1 is not None
                proba = st.norm.ppf(1 - (self.gamma / (2 * p)))
                # homoscedastic/non-robust case
                lambd = lasso_factor * s1 * np.sqrt(n) * proba

            # homoscedastic and x-dependent case
            elif self.cov_type == "nonrobust" and self.x_dependent:
                assert psi is not None
                sims = np.empty(self.n_sim)
                Xpsi = X @ np.linalg.inv(psi)
                for r in range(self.n_sim):
                    g = np.random.normal(size=(n, 1))
                    sims[r] = np.max(np.abs(np.sum(Xpsi * g, axis=0)))
                    # sims[r] = n * np.max(2 * np.abs(np.mean(Xpsi * g, axis=0)))

                lambd = lasso_factor * s1 * np.quantile(sims, 1 - self.gamma)

            # heteroscedastic/cluster robust and x-independent case
            elif self.cov_type in ("robust", "cluster") and not self.x_dependent:

                proba = st.norm.ppf(1 - (self.gamma / (2 * p)))
                # homoscedastic/non-robust case
                lambd = lasso_factor * np.sqrt(n) * proba

            # heteroscedastic/cluster robust and x-dependent case
            elif self.cov_type == "robust" and self.x_dependent:
                assert psi is not None
                sims = np.empty(self.n_sim)
                Xpsi = X @ np.linalg.inv(psi)

                for r in range(self.n_sim):
                    v = v.reshape(-1, 1)  # reshape to column vector
                    g = np.random.normal(size=(n, 1))
                    sims[r] = np.max(np.abs(np.sum(Xpsi * (v * g), axis=0)))

                lambd = lasso_factor * np.quantile(sims, 1 - self.gamma)

        return lambd

    def _cvxpy_solve(self, X, y, lambd, psi, n):

        _, p = X.shape

        beta = cp.Variable(p)
        objective = cp.Minimize(self._criterion_function(X, y, beta, lambd, psi, n))
        # define the problem
        prob = cp.Problem(objective)
        # solve the problem
        prob.solve(**self.solver_opts)
        # get fitted coefficients
        beta = beta.value
        # round coefficients to zero if they are below the tolerance
        beta[np.where(np.abs(beta) < self.zero_tol)] = 0.0

        return beta

    @staticmethod
    def _post_lasso(beta, X, y):

        nonzero_idx = np.where(beta != 0)[0]
        X_sub = X[:, nonzero_idx]
        post_beta = np.linalg.inv(X_sub.T @ X_sub) @ X_sub.T @ y
        beta[nonzero_idx] = post_beta

        return beta

    def _criterion_function(self, X, y, beta, lambd, psi, n):

        if self.sqrt:
            loss = cp.norm2(y - X @ beta) / cp.sqrt(n)
            reg = (lambd / n) * cp.norm1(psi @ beta)

        else:
            loss = cp.sum_squares(y - X @ beta) / n
            reg = (lambd / n) * cp.norm1(psi @ beta)

        return loss + reg

    def _fit(self, X, y):

        n, p = X.shape

        # set default gamma if not provided
        if self.gamma is None:
            self.gamma = 0.1 / np.log(n)

        # sqrt lasso under homoscedasticity is a one-step estimator
        if self.sqrt and self.cov_type == "nonrobust":

            psi = self._psi_calc(X=X, n=n)
            lambd = self._lambd_calc(n, p, v=None, s1=None, X=X, psi=psi)
            beta = self._cvxpy_solve(X, y, lambd, psi, n)
            if self.post:
                beta = self._post_lasso(beta, X, y)

            return {"beta": beta, "psi": psi, "lambd": lambd, "n_iter": 0}
        # calculate initial residuals based on top correlation
        r = np.empty(p)
        for k in range(p):
            r[k] = np.abs(st.pearsonr(X[:, k], y)[0])

        X_top = X[:, np.argsort(r)[-self.n_corr :]]
        beta0 = np.linalg.inv(X_top.T @ X_top) @ X_top.T @ y

        v = y - X_top @ beta0
        s1 = np.sqrt(np.mean(v**2))
        logger.info(f"Initial RMSE: {s1}")

        psi = self._psi_calc(X=X, v=v, n=n)
        lambd = self._lambd_calc(n=n, p=p, v=v, s1=s1, X=X, psi=psi)
        logger.info(f"Initial lambda: {lambd}")

        for k in range(self.max_iter):

            s0 = s1

            beta = self._cvxpy_solve(X, y, lambd, psi, n)

            if not self.lasso_psi:
                beta = self._post_lasso(beta, X, y)

            v = y - X @ beta
            s1 = np.sqrt(np.mean(v**2))

            # change in RMSE, check convergence
            if np.abs(s1 - s0) < self.convergence_tol:
                break

            psi = self._psi_calc(X=X, v=v, n=n)
            lambd = self._lambd_calc(n=n, p=p, v=v, s1=s1, X=X, psi=psi)

            if self.verbose:
                logger.info(f"Iteration {k}: RMSE: {s1}")
                logger.info(f"Iteration {k}: lambda: {lambd}")

        # final beta
        if not self.post:
            beta = self._cvxpy_solve(X, y, lambd, psi, n)
        return {"beta": beta, "psi": psi, "lambd": lambd, "n_iter": k}

    def fit(self, X, y):
        # check input
        X, y = check_X_y(X, y)

        res = self._fit(X, y)

        self.coef_ = res["beta"]
        self.psi_ = res["psi"]
        self.lambd_ = res["lambd"]
        self.n_iter_ = res["n_iter"]

        # sklearn estimator must return self
        return self

    def fit_formulat(self, formula, data):
        pass
