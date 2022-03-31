import warnings

import cvxpy as cp
import numpy as np
import numpy.linalg as la
import scipy.stats as st
import solver_fast as solver

# import solver
from patsy import dmatrices
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


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
    max_iter_shooting: int, default=10000
        Maximum number of iterations to perform in the shooting
        algorithm.
    n_sim: int, default=5000
        Number of simulations to be performed for x-dependent
        lambda calculation.
    c: float, default=1.1
        slack parameter used for lambda calculation. From
        Hansen et.al. (2020): "c needs to be greater than 1
        for the regularization event to hold asymptotically,
        but not too high as the shrinkage bias is increasing in c."
    zero_tol: float, default=1e-4
        Tolerance for the rounding of the coefficients to zero.
    convergence_tol: float, default=1e-4
        Tolerance for the convergence of the iterative estimation
        procedure.
    verbose: bool, default=False
        If True, the progress of the iterative estimation procedure
        is printed.

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
        fast=False,
        post=True,
        sqrt=False,
        fit_intercept=True,
        cov_type="nonrobust",
        x_dependent=False,
        lasso_psi=False,
        n_corr=5,
        max_iter=1,
        max_iter_shooting=1000,
        n_sim=5000,
        c=1.1,
        conv_tol=1e-4,
        zero_tol=1e-4,
        opt_tol=1e-10,
    ):
        if cov_type not in ("nonrobust", "robust", "cluster"):
            raise ValueError(
                ("cov_type must be one of 'nonrobust', 'robust', 'cluster'")
            )

        if c < 1:
            warnings.warn(
                "c should be greater than 1 for the regularization"
                " event to hold asymptotically"
            )

        self.fast = fast
        self.post = post
        self.sqrt = sqrt
        self.fit_intercept = fit_intercept
        self.cov_type = cov_type
        self.x_dependent = x_dependent
        self.lasso_psi = lasso_psi
        self.n_corr = n_corr
        self.max_iter = max_iter
        self.max_iter_shooting = max_iter_shooting
        self.n_sim = n_sim
        self.c = c
        self.conv_tol = conv_tol
        self.zero_tol = zero_tol
        self.opt_tol = opt_tol

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
                psi_1 = np.sqrt(np.mean(X**2, axis=0))
                # psi_1 = np.sqrt(Xv2 / n)
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
        # constant should be unpenalized
        if self.fit_intercept:
            psi[0] = 0.0

        return psi

    def _lambd_calc(
        self,
        n,
        p,
        gamma,
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

        if psi is not None:
            psi = np.diag(psi)

        # adjust for intercept
        if self.fit_intercept:
            p = p - 1
            if self.x_dependent:
                X = X[:, 1:]
                psi = psi[1:, 1:]

        if self.sqrt:
            lasso_factor = self.c
            # x-independent (same for robust and nonrobust)
            if not self.x_dependent:
                prob = st.norm.ppf(1 - (gamma / (2 * p)))
                lambd = lasso_factor * np.sqrt(n) * prob

            # x-dependent and nonrobust case
            elif self.x_dependent and self.cov_type == "nonrobust":
                Xpsi = X @ la.inv(psi)
                sims = np.empty(self.n_sim)
                for r in range(self.n_sim):
                    g = np.random.normal(size=(n, 1))
                    sg = np.mean(g**2)
                    sims[r] = sg * np.max(np.abs(np.sum(Xpsi * g, axis=0)))

                lambd = lasso_factor * np.quantile(sims, 1 - gamma)

            # x-dependent and robust case
            elif self.x_dependent and self.cov_type == "robust":
                Xpsi = X @ la.inv(psi)
                sims = np.empty(self.n_sim)
                for r in range(self.n_sim):
                    g = np.random.normal(size=(n, 1))
                    sg = np.mean(g**2)
                    sims[r] = sg * np.max(np.abs(np.sum(Xpsi * v[:, None] * g, axis=0)))

                lambd = lasso_factor * np.quantile(sims, 1 - gamma)

            # x-dependent and clustered case
            else:
                raise NotImplementedError(
                    "Cluster robust penalty\
                        not implemented"
                )

        else:

            lasso_factor = 2 * self.c
            # homoscedasticity and x-independent case
            if self.cov_type == "nonrobust" and not self.x_dependent:
                assert s1 is not None
                proba = st.norm.ppf(1 - (gamma / (2 * p)))
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

                lambd = lasso_factor * s1 * np.quantile(sims, 1 - gamma)

            # heteroscedastic/cluster robust and x-independent case
            elif self.cov_type in ("robust", "cluster") and not self.x_dependent:

                proba = st.norm.ppf(1 - (gamma / (2 * p)))
                lambd = lasso_factor * np.sqrt(n) * proba

            # heteroscedastic/cluster robust and x-dependent case
            elif self.cov_type == "robust" and self.x_dependent:
                sims = np.empty(self.n_sim)
                Xpsi = X @ np.linalg.inv(psi)
                for r in range(self.n_sim):
                    g = np.random.normal(size=(n, 1))
                    sims[r] = np.max(np.abs(np.sum(Xpsi * v[:, None] * g, axis=0)))

                lambd = lasso_factor * np.quantile(sims, 1 - gamma)

            # heteroscedastic/cluster robust and x-dependent case
            else:
                raise NotImplementedError(
                    "Cluster robust \
                        penalty not implemented"
                )

        return lambd

    def _starting_values(self, XX, Xy, lambd, psi):
        """Calculate starting values for the lasso."""
        if self.sqrt:
            beta_ridge = la.solve(XX + lambd * np.diag(psi**2), Xy)
        else:
            beta_ridge = la.solve(XX * 2 + lambd * np.diag(psi**2), Xy * 2)
        return beta_ridge

    @staticmethod
    def _post_lasso(beta, X, y):
        """Run post-lasso/OLS on the lasso coefficients."""

        nonzero_idx = np.where(beta != 0)[0]
        X_sub = X[:, nonzero_idx]
        post_beta = np.linalg.inv(X_sub.T @ X_sub) @ X_sub.T @ y
        beta[nonzero_idx] = post_beta

        return beta

    def _fit(self, X, y, gamma):
        """Helper function to fit the model."""

        n, p = X.shape

        # set default gamma if not provided
        if gamma is None:
            gamma = 0.1 / np.log(n)

        # intercept handling
        if self.fit_intercept:
            p += 1
            corr_range = np.arange(1, p)
            X = np.c_[np.ones(X.shape[0]), X]
        else:
            corr_range = np.arange(p)

        # precompute XX and Xy crossprods
        XX = X.T @ X
        Xy = X.T @ y

        # make matrices fortran contiguous
        XX = np.asfortranarray(XX, dtype=np.float64)
        X = np.asfortranarray(X, dtype=np.float64)
        Xy = np.asfortranarray(Xy, dtype=np.float64)
        y = np.asfortranarray(y, dtype=np.float64)

        # sqrt used under homoscedastic is one-step estimator
        if self.sqrt and self.cov_type == "nonrobust" and not self.x_dependent:

            psi = self._psi_calc(X, n)
            lambd = self._lambd_calc(n=n, p=p, gamma=gamma, X=X)
            beta_ridge = self._starting_values(XX, Xy, lambd, psi)

            beta = solver.lasso_shooting(
                X=X,
                y=y,
                XX=XX,
                Xy=Xy,
                lambd=lambd,
                psi=psi,
                starting_values=beta_ridge,
                sqrt=self.sqrt,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter_shooting,
                opt_tol=self.opt_tol,
                zero_tol=self.zero_tol,
            )
            if self.post:
                beta = self._post_lasso(beta, X, y)

            return {"beta": beta, "lambd": lambd, "psi": psi, "n_iter": 0}

        # calculate initial residuals based on top correlation
        if n > 1 and corr_range.size > 1:
            r = np.empty(p)
            for k in corr_range:
                r[k] = np.abs(st.pearsonr(X[:, k], y)[0])

            X_top = X[:, np.argsort(r)[-self.n_corr :]]
            beta0 = la.inv(X_top.T @ X_top) @ X_top.T @ y
            v = y - X_top @ beta0
        else:
            beta0 = np.ones(p) * 0.01
            v = y - X @ beta0

        s1 = np.sqrt(np.mean(v**2))

        psi = self._psi_calc(X=X, v=v, n=n)
        lambd = self._lambd_calc(
            n=n,
            p=p,
            gamma=gamma,
            v=v,
            s1=s1,
            X=X,
            psi=psi,
        )
        beta_ridge = self._starting_values(XX, Xy, lambd, psi)

        # run shooting algorithm
        beta = solver.lasso_shooting(
            X=X,
            y=y,
            XX=XX,
            Xy=Xy,
            lambd=lambd,
            psi=psi,
            starting_values=beta_ridge,
            sqrt=self.sqrt,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter_shooting,
            opt_tol=self.opt_tol,
            zero_tol=self.zero_tol,
        )

        for k in range(self.max_iter):

            s0 = s1

            # error refinement
            if not self.lasso_psi:
                beta = self._post_lasso(beta, X, y)
            v = y - X @ beta
            s1 = np.sqrt(np.mean(v**2))

            # get new estimates
            psi = self._psi_calc(X=X, v=v, n=n)
            lambd = self._lambd_calc(
                n=n,
                p=p,
                gamma=gamma,
                v=v,
                s1=s1,
                X=X,
                psi=psi,
            )
            beta = solver.lasso_shooting(
                X=X,
                y=y,
                XX=XX,
                Xy=Xy,
                lambd=lambd,
                psi=psi,
                starting_values=beta,
                sqrt=self.sqrt,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter_shooting,
                opt_tol=self.opt_tol,
                zero_tol=self.zero_tol,
            )

            if np.abs(s1 - s0) < self.conv_tol:
                break

        # end of algorithm
        if self.post:
            beta = self._post_lasso(beta, X, y)

        return {"beta": beta, "psi": psi, "lambd": lambd, "n_iter": k + 1}

    def fit(self, X, y, *, gamma=None):
        """
        Fit the model to the dataself.

        parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: array-like, shape (n_samples,)
            Target vector.
        gamma: float, optional (default: 0.1 / np.log(n_samples))
            Regularization parameter.

        returns
        -------
        self: object
            Returns self.
        """

        # check input
        X, y = check_X_y(X, y, accept_sparse=False)

        res = self._fit(X, y, gamma)
        if self.fit_intercept:
            self.intercept_ = res["beta"][0]
            self.coef_ = res["beta"][1:]
        else:
            self.coef_ = res["beta"]
            self.intercept_ = 0

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

        res = self._fit(X, y, gamma=None)

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

        pred = self.intercept_ + X @ self.coef_

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
        obj = cp.Minimize(self._criterion_function(X, y, beta, lambd, n))
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

    def _fit(self, X, y, *, gamma=None):

        n, p = X.shape

        if gamma is None:
            gamma = 0.1 / np.log(n)

        lambd0, lambd = self._lambd_calc(n, p)

        beta = self._cvxpy_solve(X, y, lambd, n, p)

        return {"beta": beta, "lambd0": lambd0, "lambd": lambd}

    def fit(self, X, y, *, gamma=None):
        """Fit the model to the data.

        parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: array-like, shape (n_samples,)
            Target vector.
        gamma: float, optional (default: 0.1 / np.log(n_samples))

        returns
        -------
        self: object
            Returns self.
        """

        # check inputs
        X, y = check_X_y(X, y, accept_sparse=True, ensure_2d=True)

        # assert y is binary
        if np.unique(y).shape[0] != 2:
            raise ValueError("y must be binary")

        res = self._fit(X, y, gamma=gamma)

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
