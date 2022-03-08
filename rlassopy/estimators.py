import logging

import cvxpy as cp
import numpy as np
import scipy.stats as st
from patsy import dmatrices
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Rlasso(BaseEstimator, RegressorMixin):
    """
    Rigorous Lasso estimator with theoretically justified
    penalty levels and desirable convergence properties.

    Parameters
    ----------
    post: bool, default=True
        If True, post-lasso is used to estimate betas.
    sqrt: bool, default=False
        If True, sqrt lasso criterion is minimized:
        loss = ||y - X @ beta||_2 / sqrt(n)
        see: Belloni, A., Chernozhukov, V., & Wang, L. (2011).
        Square-root lasso: pivotal recovery of sparse signals via
        conic programming. Biometrika, 98(4), 791-806.
    fit_intercept: bool, default=True
        If True, intercept is estimated.
    cov_type: str, default="nonrobust"
        Type of covariance matrix.
        "nonrobust" - nonrobust covariance matrix
        "robust" - robust covariance matrix
        "cluster" - cluster robust covariance matrix
    x_dependent: bool, default=False
        If True, the less conservative lambda is estimated
        by simulation using the conditional distribution of the
        design matrix.
    lasso_psi: bool, default=False
        If True, post-lasso is not used to obtain the residuals
        during the iterative estimation procedure.
    n_corr: int, default=5
        Number of correlated variables to be used in the
        for initial calculation of the residuals.
    max_iter: int, default=2
        Maximum number of iterations to perform in the iterative
        estimation procedure.
    n_sim: int, default=5000
        Number of simulations to be performed for x-dependent
        lambda calculation.
    c: float, default=1.1
        slack parameter used for lambda calculation. From
        Hansen et.al. (2020): "c needs to be greater than 1
        for the regularization event to hold asymptotically,
        but not too high as the shrinkage bias is increasing in c."
    gamma: float, default=0.1 / log(n)
        Significance level for the quantile function in lambda
        calculation. Probability of the regularization event to
        hold asymptotically = 1 - gamma.
    zero_tol: float, default=1e-4
        Tolerance for the rounding of the coefficients to zero.
    convergence_tol: float, default=1e-4
        Tolerance for the convergence of the iterative estimation
        procedure.
    verbose: bool, default=False
        If True, the progress of the iterative estimation procedure
        is printed.
    solver_opts: dict, default=None
        Dictionary with additional options for the cvxpy solver.
        See cvxpy documentation for details: https://cvxpy.org/.

    attributes
    ----------
    coef_: numpy.array, shape (n_features,)
        Estimated coefficients.
    intercept_: float
        Estimated intercept.
    lambd_: float
        Estimated lambda/overall penalty level.
    psi_: numpy.array, shape (n_features, n_features)
        Estimated penalty loadings.
    n_iter_: int
        Number of iterations performed by the rlasso algorithm.
    endog_: str
        Name of the endogenous variable. Only stored if
        fit_formula method is used.
    exog_: list[str]
        Names of the exogenous variables. Only stored if
        fit_formula method is used.
    """

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

        """Calculate the penalty loadings."""

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
        """Calculate the lambda/overall penalty level."""

        # TODO Always return both lambda and lambda scaled by RMSE
        # for the purpose of comparison between specifications.

        # TODO: Implement cluster robust case

        # catch parameters are provided
        # if self.cov_type == "nonrobust" and s1 is None:
        #     raise ValueError(f"RMSE must be provided for {self.cov_type}")
        #
        # if self.x_dependent and psi is None:
        #     raise ValueError("X must be provided for x_dependent")

        if self.sqrt:
            lasso_factor = self.c
            # x-independent (same for robust and nonrobust)
            if not self.x_dependent:
                prob = st.norm.ppf(1 - (self.gamma / (2 * p)))
                lambd = lasso_factor * np.sqrt(n) * prob

            # x-dependent and nonrobust case
            elif self.x_dependent and self.cov_type == "nonrobust":
                Xpsi = X @ np.linalg.inv(psi)
                sims = np.empty(self.n_sim)
                for r in range(self.n_sim):
                    g = np.random.normal(size=(n, 1))
                    sg = np.mean(g**2)
                    sims[r] = sg * np.max(np.abs(np.sum(Xpsi * g, axis=0)))

                lambd = lasso_factor * np.quantile(sims, 1 - self.gamma)

            # x-dependent and robust case
            elif self.x_dependent and self.cov_type == "robust":
                Xpsi = X @ np.linalg.inv(psi)
                v = v.reshape(-1, 1)
                sims = np.empty(self.n_sim)
                for r in range(self.n_sim):
                    g = np.random.normal(size=(n, 1))
                    sg = np.mean(g**2)
                    sims[r] = sg * np.max(np.abs(np.sum(Xpsi * v * g, axis=0)))

                lambd = lasso_factor * np.quantile(sims, 1 - self.gamma)

            # x-dependent and clustered case
            else:
                raise NotImplementedError("Cluster robust penalty not implemented")

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
                lambd = lasso_factor * np.sqrt(n) * proba

            # heteroscedastic/cluster robust and x-dependent case
            elif self.cov_type == "robust" and self.x_dependent:
                sims = np.empty(self.n_sim)
                Xpsi = X @ np.linalg.inv(psi)
                v = v.reshape(-1, 1)  # reshape to column vector
                for r in range(self.n_sim):
                    g = np.random.normal(size=(n, 1))
                    sims[r] = np.max(np.abs(np.sum(Xpsi * v * g, axis=0)))

                lambd = lasso_factor * np.quantile(sims, 1 - self.gamma)

            # heteroscedastic/cluster robust and x-dependent case
            else:
                raise NotImplementedError("Cluster robust penalty not implemented")

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
        """Run post-lasso/OLS on the lasso coefficients."""

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
        """Helper function to fit the model."""

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
        """
        Fit the model to the dataself.

        parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: array-like, shape (n_samples,)
            Target vector.

        returns
        -------
        self: object
            Returns self.
        """

        # check input
        X, y = check_X_y(X, y)

        res = self._fit(X, y)

        self.coef_ = res["beta"]
        self.psi_ = res["psi"]
        self.lambd_ = res["lambd"]
        self.n_iter_ = res["n_iter"]

        # sklearn estimator must return self
        return self

    def fit_formula(self, formula, data):
        """
        Fit the the model to the data using fomula language.

        Parameters
        ----------
        formula: str
            Formula to fit the model. Ex: "y ~ x1 + x2 + x3"
        data: Union[pandas.DataFrame, numpy.recarray, dict]
            Dataset to fit the model.

        Returns
        -------
        self: object
            Returns self.
        """

        # TODO: Solution to handle intercept

        y, X = dmatrices(formula, data)

        self.endog_ = y.design_info.column_names[0]
        self.exog_ = X.design_info.column_names

        X, y = np.asarray(X), np.asarray(y)
        y = y.flatten()

        res = self._fit(X, y)

        self.coef_ = res["beta"]
        self.psi_ = res["psi"]
        self.lambd_ = res["lambd"]
        self.n_iter_ = res["n_iter"]

        # sklearn estimator must return self
        return self

    def predict(self, X):

        """
        Use fitted model to predict on new data.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Design matrix.

        returns
        -------
        y_pred: numpy.array, shape (n_samples,)
            Predicted target values.
        """

        # check if fitted
        check_is_fitted(self, ["coef_"])
        X = check_array(X)

        pred = X @ self.coef_

        if hasattr(self, "intercept_"):
            pred += self.intercept_

        return pred


class RlassoLogit(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        post=True,
        fit_intercept=True,
        c=1.1,
        gamma=None,
        zero_tol=1e-4,
        solver_opts=None,
    ):
        """Rigorous Lasso Logistic Regression."""

        self.post = post
        self.fit_intercept = fit_intercept
        self.c = c
        self.gamma = gamma
        self.zero_tol = zero_tol
        self.solver_opts = solver_opts or {}

    def _criterion_function(self, X, y, beta, lambd, n, regularization=True):
        """Criterion function for the penalized Lasso Logistic Regression."""

        ll = cp.sum(cp.multiply(y, X @ beta) - cp.logistic(X @ beta)) / n
        if not regularization:
            return -ll
        else:
            reg = (lambd / n) * cp.norm1(beta)
            return -(ll - reg)

    def _cvxpy_solve(self, X, y, lambd, n, p):
        """Solve the problem using cvxpy."""

        beta = cp.Variable(p)
        objective = cp.Minimize(self._criterion_function(X, y, beta, lambd, n))
        prob = cp.Problem(obj)

        # solve problem and return beta
        prob.solve(**self.solver_opts)
        beta = beta.value
        beta[np.abs(beta) < self.zero_tol] = 0.0

        return beta

    def _decision_function(self, X, beta):
        """Compute the decision function of the model."""
        return 1 / (1 + np.exp(-X @ beta))

    def _lambd_calc(self, n, p):
        lambd0 = (self.c / 2) * np.sqrt(n) * st.norm.ppf(1 - self.gamma / (2 * p))
        lambd = lambd0 / (2 * n)
        return lambd0, lambd

    def _fit(self, X, y):

        n, p = X.shape

        if self.gamma is None:
            self.gamma = 0.1 / np.log(n)

        lambd0, lambd = self._lambd_calc(n, p)

        beta = self._cvxpy_solve(X, y, lambd, n, p)

        return {"beta": beta, "lambd0": lambd0, "lambd": lambd}

    def fit(self, X, y):
        """Fit the model to the data."""

        # check inputs
        X, y = check_X_y(X, y, accept_sparse=True, ensure_2d=True)

        # assert y is binary
        if np.unique(y).shape[0] != 2:
            raise ValueError("y must be binary")

        res = self._fit(X, y)

        self.coef_ = res["beta"]
        self.lambd0_ = res["lambd0"]
        self.lambd_ = res["lambd"]

        return self

    def predict(self, X):
        """Predict the class labels for X."""
        # check model is fitted and inputs are correct
        check_is_fitted(self, ["coef_"])
        X = check_array(X)

        probas = self._decision_function(X, self.coef_)
        return np.where(probas > 0.5, 1, 0)

    def predict_proba(self, X):
        """Predict class probabilities for X."""
        # check model is fitted and inputs are correct
        check_is_fitted(self, ["coef_"])
        X = check_array(X)

        return self._decision_function(X, self.coef_)
