import warnings

import cvxpy as cp
import numpy as np
import numpy.linalg as la
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from _solver_fast import _cd_solver

# import solver
from patsy import dmatrices
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_random_state,
    check_X_y,
)
from statsmodels.iolib import SimpleTable


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
        If True, unpenalized intercept is estimated.

    cov_type: str, default="nonrobust"
        Type of covariance matrix.
        "nonrobust" - nonrobust covariance matrix
        "robust" - robust covariance matrix
        "cluster" - cluster robust covariance matrix

    x_dependent: bool, default=False
        If True, the less conservative lambda is estimated
        by simulation using the conditional distribution of the
        design matrix.

    n_sim: int, default=5000
        Number of simulations to be performed for x-dependent
        lambda calculation.

    random_state: int, default=None
        Random seed used for simulations if `x_dependent` is True.

    lasso_psi: bool, default=False
        If True, post-lasso is not used to obtain the residuals
        during the iterative estimation procedure.

    n_corr: int, default=5
        Number of correlated variables to be used in the
        for initial calculation of the residuals.

    c: float, default=1.1
        slack parameter used for lambda calculation. From
        Hansen et.al. (2020): "c needs to be greater than 1
        for the regularization event to hold asymptotically,
        but not too high as the shrinkage bias is increasing in c."

    gamma: float, optional=None
        Regularization parameter. If not provided
        gamma is calculated as 0.1 / np.log(n_samples)

    max_iter: int, default=2
        Maximum number of iterations to perform in the iterative
        estimation procedure.

    conv_tol: float, default=1e-4
        Tolerance for the convergence of the iterative estimation
        procedure.

    solver: str, default="cd"
        Solver to be used for the iterative estimation procedure.
        "cd" - coordinate descent
        "cvxpy" - cvxpy solver

    cd_max_iter: int, default=10000
        Maximum number of iterations to perform in the shooting
        algorithm.

    cd_tol: float, default=1e-10
        Tolerance for the coordinate descent algorithm.

    cvxpy_opts: dict, default=None
        Options to be passed to the cvxpy solver. See cvxpy documentation
        for more details:
        https://www.cvxpy.org/tutorial/advanced/index.html#solve-method-options

    zero_tol: float, default=1e-4
        Tolerance for the rounding of the coefficients to zero.

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

    n_features_in_: int
        Number of features in the input data.

    n_samples_: int
        Number of samples/observations in the input data.

    feature_names_in_: str
        Name of the endogenous variable. Only stored if
        fit_formula method is used.

    outcome_name_in_: list[str]
        Name of the exogenous variables. Only stored if
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
        random_state=None,
        lasso_psi=False,
        prestd=False,
        n_corr=5,
        max_iter=2,
        conv_tol=1e-4,
        n_sim=5000,
        c=1.1,
        gamma=None,
        solver="cd",
        cd_max_iter=1000,
        cd_tol=1e-10,
        cvxpy_opts=None,
        zero_tol=1e-4,
    ):

        self.post = post
        self.sqrt = sqrt
        self.fit_intercept = fit_intercept
        self.cov_type = cov_type
        self.x_dependent = x_dependent
        self.random_state = random_state
        self.lasso_psi = lasso_psi
        self.prestd = prestd
        self.n_corr = n_corr
        self.max_iter = max_iter
        self.conv_tol = conv_tol
        self.n_sim = n_sim
        self.c = c
        self.gamma = gamma
        self.solver = solver
        self.cd_max_iter = cd_max_iter
        self.cd_tol = cd_tol
        self.zero_tol = zero_tol
        self.cvxpy_opts = cvxpy_opts

    def _psi_calc(self, X, n, v=None):

        """Calculate the penalty loadings."""

        # TODO Implement cluster robust covariance
        # if prestandardized X, set loadings to ones
        if self.prestd:
            psi = np.ones(self.n_features_in_)

        # sqrt case
        elif self.sqrt:

            if self.cov_type == "nonrobust":
                psi = np.sqrt(np.mean(X**2, axis=0))

            # heteroscedastic robust case
            elif self.cov_type == "robust" and v is not None:
                Xv2 = np.einsum("ij, i -> j", X**2, v**2)
                psi_1 = np.sqrt(np.mean(X**2, axis=0))
                psi_2 = np.sqrt(Xv2 / np.sum(v**2))
                psi = np.maximum(psi_1, psi_2)
            # clustered
            else:
                raise NotImplementedError(
                    "Cluster robust loadings not \
                                                implemented"
                )

        elif self.cov_type == "nonrobust":
            psi = np.sqrt(np.mean(X**2, axis=0))

        elif self.cov_type == "robust" and v is not None:
            Xe2 = np.einsum("ij, i -> j", X**2, v**2)
            psi = np.sqrt(Xe2 / n)

        else:
            raise NotImplementedError(
                "Cluster robust loadings not \
                                                implemented"
            )

        return psi

    def _lambd_calc(
        self,
        n,
        p,
        X,
        *,
        v=None,
        s1=None,
        psi=None,
    ):  # sourcery skip: remove-redundant-if
        """Calculate the lambda/overall penalty level."""

        # TODO Always return both lambda and lambda scaled by RMSE
        # for the purpose of comparison between specifications.

        # TODO: Implement cluster robust case

        # empirical gamma if not provided
        gamma = self.gamma or 0.1 / np.log(n)
        if psi is not None:
            psi = np.diag(psi)

        if self.sqrt:
            lf = self.c
            # x-independent (same for robust and nonrobust)
            if not self.x_dependent:
                prob = st.norm.ppf(1 - (gamma / (2 * p)))
                lambd = lf * np.sqrt(n) * prob

            elif self.cov_type == "nonrobust":
                Xpsi = X @ la.inv(psi)
                sims = np.empty(self.n_sim)
                for r in range(self.n_sim):
                    g = self.random_state_.normal(size=(n, 1))
                    sg = np.mean(g**2)
                    sims[r] = sg * np.max(np.abs(np.sum(Xpsi * g, axis=0)))

                lambd = lf * np.quantile(sims, 1 - gamma)

            elif self.cov_type == "robust":
                Xpsi = X @ la.inv(psi)
                sims = np.empty(self.n_sim)
                for r in range(self.n_sim):
                    g = self.random_state_.normal(size=(n, 1))
                    sg = np.mean(g**2)
                    sims[r] = sg * np.max(np.abs(np.sum(Xpsi * v[:, None] * g, axis=0)))

                lambd = lf * np.quantile(sims, 1 - gamma)

            else:
                raise NotImplementedError(
                    "Cluster robust penalty\
                        not implemented"
                )

        else:

            lf = 2 * self.c
            # homoscedasticity and x-independent case
            if self.cov_type == "nonrobust" and not self.x_dependent:
                assert s1 is not None
                proba = st.norm.ppf(1 - (gamma / (2 * p)))
                # homoscedastic/non-robust case
                lambd = lf * s1 * np.sqrt(n) * proba

            elif self.cov_type == "nonrobust" and self.x_dependent:
                assert psi is not None
                sims = np.empty(self.n_sim)
                Xpsi = X @ la.inv(psi)
                for r in range(self.n_sim):
                    g = self.random_state_.normal(size=(n, 1))
                    sims[r] = np.max(np.abs(np.sum(Xpsi * g, axis=0)))

                lambd = lf * s1 * np.quantile(sims, 1 - gamma)

            # heteroscedastic/cluster robust and x-independent case
            elif self.cov_type in ("robust", "cluster") and not self.x_dependent:

                proba = st.norm.ppf(1 - (gamma / (2 * p)))
                lambd = lf * np.sqrt(n) * proba

            # heteroscedastic/cluster robust and x-dependent case
            elif self.cov_type == "robust" and self.x_dependent:
                sims = np.empty(self.n_sim)
                Xpsi = X @ la.inv(psi)
                for r in range(self.n_sim):
                    g = self.random_state_.normal(size=(n, 1))
                    sims[r] = np.max(np.abs(np.sum(Xpsi * v[:, None] * g, axis=0)))

                lambd = lf * np.quantile(sims, 1 - gamma)

            # heteroscedastic/cluster robust and x-dependent case
            else:
                raise NotImplementedError(
                    "Cluster robust \
                        penalty not implemented"
                )

        return lambd

    def _cvxpy_solver(
        self,
        X,
        y,
        lambd,
        psi,
        n,
        p,
    ):
        """
        Solve the lasso problem using cvxpy
        """

        beta = cp.Variable(p)

        if self.sqrt:
            loss = cp.norm2(y - X @ beta) / cp.sqrt(n)
        else:
            loss = cp.sum_squares(y - X @ beta) / n

        reg = (lambd / n) * cp.norm1(np.diag(psi) @ beta)
        objective = cp.Minimize(loss + reg)
        prob = cp.Problem(objective)
        prob.solve(**self.cvxpy_opts or {})

        # round beta to zero if below threshold
        beta = beta.value
        beta[np.abs(beta) < self.zero_tol] = 0.0

        return beta

    def _OLS(self, X, y):
        """
        Solve the OLS problem
        """
        # add dim if X is 1-d
        if X.ndim == 1:
            X = X[:, None]
        try:
            return la.solve(X.T @ X, X.T @ y)
        except la.LinAlgError:
            warnings.warn("Singular matrix encountered. invoking lstsq solver for OLS")
            return la.lstsq(X, y, rcond=None)[0]

    def _post_lasso(self, beta, X, y):
        """Replace the non-zero lasso coefficients by OLS."""

        nonzero_idx = np.where(beta != 0)[0]
        X_sub = X[:, nonzero_idx]
        post_beta = self._OLS(X_sub, y)
        beta[nonzero_idx] = post_beta

        return beta

    def _starting_values(self, XX, Xy, lambd, psi):
        """Calculate starting values for the lasso."""
        if self.sqrt:
            return la.solve(XX + lambd * np.diag(psi**2), Xy)
        else:
            return la.solve(XX * 2 + lambd * np.diag(psi**2), Xy * 2)

    def _fit(self, X, y, *, partial=None, cluster_var=None):
        """Helper function to fit the model."""

        if self.max_iter < 0:
            raise ValueError("`max_iter` cannot be negative")

        if self.cov_type not in ("nonrobust", "robust", "cluster"):
            raise ValueError(
                ("cov_type must be one of 'nonrobust', 'robust', 'cluster'")
            )

        if self.solver not in ("cd", "cvxpy"):
            raise ValueError("solver must be one of 'cd', 'cvxpy'")

        if self.c < 1:
            warnings.warn(
                "c should be greater than 1 for the regularization"
                " event to hold asymptotically"
            )

        if self.cov_type == "cluster" and cluster_var is None:
            raise ValueError(
                "cluster_vars must be specified for cluster robust penalty"
            )

        X, y = check_X_y(X, y, accept_sparse=False, ensure_min_samples=2)

        p = self.n_features_in_ = X.shape[1]
        n = X.shape[0]

        # check random state
        self.random_state_ = check_random_state(self.random_state)

        # intercept and pre-standardization handling
        if self.fit_intercept or self.prestd:
            X_mean, y_mean = np.mean(X, axis=0), np.mean(y)
            X, y = X - X_mean, y - y_mean

        if self.prestd:
            X_std, y_std = np.std(X, axis=0), np.std(y)
            X, y = X / X_std, y / y_std

        # pre-allocate arrays for coordinate descent solver
        if self.solver == "cd":
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
            lambd = self._lambd_calc(n=n, p=p, X=X)

            if self.solver == "cd":
                beta_ridge = self._starting_values(XX, Xy, lambd, psi)
                beta = _cd_solver(
                    X=X,
                    y=y,
                    XX=XX,
                    Xy=Xy,
                    lambd=lambd,
                    psi=psi,
                    starting_values=beta_ridge,
                    sqrt=self.sqrt,
                    fit_intercept=self.fit_intercept,
                    max_iter=self.cd_max_iter,
                    opt_tol=self.cd_tol,
                    zero_tol=self.zero_tol,
                )
            else:
                beta = self._cvxpy_solver(
                    X=X,
                    y=y,
                    lambd=lambd,
                    psi=psi,
                    n=n,
                    p=p,
                )

            if self.post:
                beta = self._post_lasso(beta, X, y)

            # rescale beta
            if self.prestd:
                beta *= y_std / X_std

            self.intercept_ = y_mean - X_mean @ beta if self.fit_intercept else 0.0
            self.nonzero_idx_ = np.where(beta != 0)[0]
            self.coef_ = beta
            self.n_iter_ = 1
            self.lambd_ = lambd
            self.psi_ = psi

            return

        # calculate error based on initial
        # highly correlated vars
        r = np.empty(p)
        for k in range(p):
            r[k] = np.abs(st.pearsonr(X[:, k], y)[0])

        X_top = X[:, np.argsort(r)[-self.n_corr :]]
        beta0 = self._OLS(X_top, y)
        v = y - X_top @ beta0
        s1 = np.sqrt(np.mean(v**2))

        psi = self._psi_calc(X=X, v=v, n=n)
        lambd = self._lambd_calc(
            n=n,
            p=p,
            v=v,
            s1=s1,
            X=X,
            psi=psi,
        )

        # get initial estimates k=0
        if self.solver == "cd":
            beta_ridge = self._starting_values(XX, Xy, lambd, psi)
            beta = _cd_solver(
                X=X,
                y=y,
                XX=XX,
                Xy=Xy,
                lambd=lambd,
                psi=psi,
                starting_values=beta_ridge,
                sqrt=self.sqrt,
                fit_intercept=self.fit_intercept,
                max_iter=self.cd_max_iter,
                opt_tol=self.cd_tol,
                zero_tol=self.zero_tol,
            )

        else:
            beta = self._cvxpy_solver(
                X=X,
                y=y,
                lambd=lambd,
                psi=psi,
                n=n,
                p=p,
            )

        for k in range(self.max_iter):

            s0 = s1

            # post lasso handling
            if not self.lasso_psi:
                beta = self._post_lasso(beta, X, y)

            # error refinement
            v = y - X @ beta
            s1 = np.sqrt(np.mean(v**2))

            # if convergence not reached get new estimates of lambd and psi
            psi = self._psi_calc(X=X, v=v, n=n)
            lambd = self._lambd_calc(
                n=n,
                p=p,
                v=v,
                s1=s1,
                X=X,
                psi=psi,
            )

            if self.solver == "cd":
                beta = _cd_solver(
                    X=X,
                    y=y,
                    XX=XX,
                    Xy=Xy,
                    lambd=lambd,
                    psi=psi,
                    starting_values=beta_ridge,
                    sqrt=self.sqrt,
                    fit_intercept=self.fit_intercept,
                    max_iter=self.cd_max_iter,
                    opt_tol=self.cd_tol,
                    zero_tol=self.zero_tol,
                )

            else:
                beta = self._cvxpy_solver(
                    X=X,
                    y=y,
                    lambd=lambd,
                    psi=psi,
                    n=n,
                    p=p,
                )

            # check convergence
            if np.abs(s1 - s0) < self.conv_tol:
                break
        # end of algorithm
        if self.post and not self.lasso_psi:
            beta = self._post_lasso(beta, X, y)

        # rescale beta if standardized
        if self.prestd:
            beta *= y_std / X_std

        self.intercept_ = y_mean - X_mean @ beta if self.fit_intercept else 0.0
        self.nonzero_idx_ = np.where(beta != 0)[0]
        self.coef_ = beta
        self.n_iter_ = k + 1 if self.max_iter > 0 else 1
        self.lambd_ = lambd
        self.psi_ = psi

    def fit(self, X, y, *, partial=None):
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

        self._fit(X, y)

        # sklearn estimator must return self
        return self

    def fit_formula(self, formula, data, *, partial=None):
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

        self.feature_names_in_ = X.design_info.column_names
        self.outcome_name_in_ = y.design_info.column_names

        X, y = np.asarray(X), np.asarray(y)
        y = y.flatten()
        # check if intercept is in data
        if "Intercept" in self.feature_names_in_:
            if not self.fit_intercept:
                raise ValueError(
                    (
                        "Intercept is in data but fit_intercept is False."
                        " Set fit_intercept to True to fit intercept or"
                        " update the formula to remove the intercept"
                    )
                )
            # drop column of ones from X
            # since intercept calculated in _fit
            # by partialing out
            X = X[:, 1:]

        self._fit(X, y)
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
        check_is_fitted(self)
        X = check_array(X)

        return self.intercept_ + X @ self.coef_


class RlassoLogit(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        post=True,
        fit_intercept=True,
        c=1.1,
        gamma=0.05,
        zero_tol=1e-4,
        solver_opts=None,
    ):
        """Rigorous Lasso Logistic Regression."""

        self.post = post
        self.fit_intercept = fit_intercept
        self.c = c
        self.gamma = gamma
        self.zero_tol = zero_tol
        self.solver_opts = solver_opts

    def _criterion_function(self, X, y, beta, lambd, n, regularization=True):
        """Criterion function for the penalized Lasso Logistic Regression."""

        ll = cp.sum(cp.multiply(y, X @ beta) - cp.logistic(X @ beta)) / n
        if not regularization:
            return -ll
        reg = (lambd / n) * cp.norm1(beta)
        return -(ll - reg)

    def _cvxpy_solve(self, X, y, lambd, n, p):
        """Solve the problem using cvxpy."""

        beta = cp.Variable(p)
        obj = cp.Minimize(self._criterion_function(X, y, beta, lambd, n))
        prob = cp.Problem(obj)

        # solve problem and return beta
        prob.solve(**self.solver_opts or {})
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
        check_is_fitted(self)
        X = check_array(X)

        return self._decision_function(X, self.coef_)

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X."""
        return np.log(self._decision_function(X, self.coef_))


class RlassoEffect:

    """
    Rigorous Lasso IV Regression.
    """

    def __init__(
        self,
        *,
        method="partialling-out",
        significance_lvl=0.05,
        post=True,
        sqrt=False,
        fit_intercept=True,
        cov_type="nonrobust",
        x_dependent=False,
        random_state=None,
        lasso_psi=False,
        prestd=False,
        n_corr=5,
        max_iter=2,
        conv_tol=1e-4,
        n_sim=5000,
        c=1.1,
        gamma=None,
        solver="cd",
        cd_max_iter=1000,
        cd_tol=1e-10,
        cvxpy_opts=None,
        zero_tol=1e-4,
    ):

        self.method = method
        self.significance_lvl = significance_lvl
        self.post = post
        self.sqrt = sqrt
        self.fit_intercept = fit_intercept
        self.cov_type = cov_type
        self.x_dependent = x_dependent
        self.random_state = random_state
        self.lasso_psi = lasso_psi
        self.prestd = prestd
        self.n_corr = n_corr
        self.n_sim = n_sim
        self.c = c
        self.gamma = gamma
        self.max_iter = max_iter
        self.conv_tol = conv_tol
        self.solver = solver
        self.cd_max_iter = cd_max_iter
        self.cd_tol = cd_tol
        self.zero_tol = zero_tol
        self.cvxpy_opts = cvxpy_opts

        self.rlasso = Rlasso(
            post=post,
            sqrt=sqrt,
            fit_intercept=fit_intercept,
            cov_type=cov_type,
            x_dependent=x_dependent,
            random_state=random_state,
            lasso_psi=lasso_psi,
            prestd=prestd,
            n_corr=n_corr,
            n_sim=n_sim,
            c=c,
            gamma=gamma,
            max_iter=max_iter,
            conv_tol=conv_tol,
            solver=solver,
            cd_max_iter=cd_max_iter,
            cd_tol=cd_tol,
            cvxpy_opts=cvxpy_opts,
        )

    def _check_input(self, X, idx):

        # check if str or int repass
        # as list of str or int
        if isinstance(idx, (int, str)):
            self._check_input(X, [idx])
        if isinstance(idx, list):
            # check if elements are str or int
            if all(isinstance(x, str) for x in idx):
                if not isinstance(X, pd.DataFrame):
                    raise ValueError(
                        "X must be a pandas DataFrame if index is a list of str"
                    )
                # check if all str are in X
                if any(x not in X.columns for x in idx):
                    raise ValueError("Index must be a subset of DataFrame.columns")

                exog_names = idx
                control_names = [x for x in X.columns if x not in idx]
                # get positions of exog_names in X
                idx = [X.columns.get_loc(x) for x in idx]

            elif all(isinstance(x, int) for x in idx):
                # check that x are columns in X
                if any(x not in range(X.shape[1]) for x in idx):
                    raise ValueError("Index not in range of X")

                exog_names = [f"x{k+1}" for k in idx]
                control_names = [f"x{k+1}" for k in range(X.shape[1]) if k not in idx]

            else:
                raise ValueError("Index must be a list of str or int")

        return exog_names, control_names

    def _fit(self, X, y, idx):

        # check inputs
        self.exog_names_, self.control_names = self._check_input(X, idx)
        self.dep_name_ = "y"
        X, y = check_X_y(X, y)

        if self.method not in ("double-selection", "partialling-out"):
            raise ValueError(
                "Selection method must be 'double-selection' or 'partialling-out'"
            )

        n, p = X.shape
        self.n_obs_ = n
        self.p_exog_ = len(idx)
        self.p_controls_ = p - self.p_exog_

        self.coef_ = np.empty(self.p_exog_)
        self.se_ = np.empty(self.p_exog_)
        self.t_val_ = np.empty(self.p_exog_)
        self.p_val_ = np.empty(self.p_exog_)
        self.ci_ = np.empty((self.p_exog_, 2))

        for i, j in enumerate(idx):

            Z = np.delete(X, j, axis=1)
            d = X[:, j]

            res = self._est_single_effect(Z, y, d)

            self.coef_[i] = res["alpha"]
            self.se_[i] = res["se"]
            self.t_val_[i] = res["t"]
            self.p_val_[i] = res["p"]
            self.ci_[i] = res["ci"]

    def _est_single_effect(self, X, y, d):

        if self.method == "partialing-out":
            lasso1 = self.rlasso.fit(X, y)
            yr = y - lasso1.predict(X)
            lasso2 = self.rlasso.fit(X, d)
            dr = d - lasso2.predict(X)

            ols = sm.OLS(exog=dr, endog=yr).fit()
            alpha = ols.params[0]
            s = np.var(yr - ols.predict(dr)) / np.sum((dr - dr.mean()) ** 2)
            se = np.sqrt(s)
            t = alpha / se

        else:
            I1 = self.rlasso.fit(X, d).nonzero_idx_
            I2 = self.rlasso.fit(X, y).nonzero_idx_
            I = np.union1d(I1, I2)

            X = np.c_[d, X[:, I]]
            reg1 = sm.OLS(y, X).fit()
            alpha = reg1.params[0]
            e = y - reg1.predict(X)
            Xi = e * np.sqrt(self.n_obs_ / (self.n_obs_ - I.size - 1))

            reg2 = sm.OLS(d, X[:, 1:]).fit()
            v = d - reg2.predict(X[:, 1:])

            var = (1 / np.mean(v**2)) ** 2 * np.mean(v**2 * Xi**2) / self.n_obs_

        se = np.sqrt(var)
        t = alpha / se
        p = 2 * st.norm.ppf(-np.abs(t))
        # if nan, set to 0
        p = 0.0 if np.isnan(p) else p
        q = se * st.norm.ppf(np.abs(1 - (self.significance_lvl / 2)))
        ci = np.array([alpha - q, alpha + q])

        return {"alpha": alpha, "se": se, "t": t, "p": p, "ci": ci}

    def fit(self, X, y, d_idx):
        """
        Fit the model to the data.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: array-like, shape (n_samples,)
            Target vector.
        d_idx:
        gamma: float, optional (default: 0.1 / np.log(n_samples))

        Returns
        -------
        """

        self._fit(X, y, d_idx)

        return self

    def summary(self):
        """
        Return a summary of the model.
        """

        check_is_fitted(self, "coef_")

        from tabulate import tabulate

        info_tbl = [
            ["Dep. Variable:", "y"],
            ["Model:", "sqrt-rlasso" if self.sqrt else "rlasso"],
            ["Inference method:", self.method],
            ["No. observations:", self.n_obs_],
            ["No. controls:", self.p_controls_],
        ]

        # data table
        headers = ["", "coef", "std err", "t", "P>|t|", "[0.025", "0.975]"]
        data = np.c_[self.coef_, self.se_, self.t_val_, self.p_val_, self.ci_].tolist()
        data_tbl = tabulate(
            data, headers=headers, showindex=self.exog_names_, tablefmt="fancy_grid"
        )  # .split_lines()

        # set title and notes
        tbl_width = len(data_tbl.splitlines()[0])
        title = "REGRESSION RESULTS".center(tbl_width)

        tab2 = tabulate(info_tbl, tablefmt="fancy_grid").center(tbl_width)

        note1 = "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified."
        # print res
        print(title, tab2, data_tbl, note1, sep="\n")
