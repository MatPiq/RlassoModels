import warnings

import cvxpy as cp
import numpy as np
import numpy.linalg as la
import pandas as pd
import scipy.stats as st
from _solver_fast import _cd_solver
from linearmodels.iv import IV2SLS, compare
from patsy import dmatrices
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_random_state,
    check_X_y,
)
from statsmodels.api import add_constant


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

    prestd: bool, default=False
        If True, the data is prestandardized instead of
        on the fly by penalty loadings. Currently only
        supports homoscedastic case.

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

    Attributes
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
        the input data is a pandas dataframe.

    References
    ----------
    Belloni, A., & Chernozhukov, V. (2013). Least squares after model selection
        in high-dimensional sparse models. Bernoulli, 19(2), 521-547.

    Belloni, A., Chernozhukov, V., & Wang, L. (2011).
        Square-root lasso: pivotal recovery of sparse signals via conic programming.
        Biometrika, 98(4), 791-806.

    Ahrens, A., Hansen, C. B., & Schaffer, M. E. (2020). lassopack: Model
        selection and prediction with regularized regression in Stata.
        The Stata Journal, 20(1), 176-235.

    Chernozhukov, V., Hansen, C., & Spindler, M. (2016).
        hdm: High-dimensional metrics. arXiv preprint arXiv:1608.00354.
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

        if self.cov_type == "cluster" and cluster_var is None:
            raise ValueError(
                "cluster_vars must be specified for cluster robust penalty"
            )

        if self.c < 1:
            warnings.warn(
                "c should be greater than 1 for the regularization"
                " event to hold asymptotically"
            )

        if self.prestd and self.cov_type in ("robust", "cluster"):
            warnings.warn(
                "prestd is not implemented for robust and cluster robust penalty. "
                "Data is assumed to be homoscedastic."
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
        # store feature names if dataset is pandas
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns

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

        y, X = dmatrices(formula, data)

        self.feature_names_in_ = X.design_info.column_names

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


class RlassoIV:
    """
    Rigorous Lasso for instrumental-variable estimation in
    the presence of high-dimensional instruments and/or
    controls. Uses the post-double-selection (PDS) and
    post-regularization (CHS) methods for estimation, see
    references below.

    Parameters
    ----------
    select_X: bool, optional (default: True)
        Whether to use lasso/post-lasso for feature
        selection of high-dim controls.

    select_Z: bool, optional (default: True)
        Whether to use lasso/post-lasso for feature
        selection of high-dim instruments.

    post: bool, default=True
        If True, post-lasso is used to estimate betas.
        Note that `post` will only affect the results
        for the post-regularization (CHS) method
        and not those of post-double-selection (pds).

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

    prestd: bool, default=False
        If True, the data is prestandardized instead of
        on the fly by penalty loadings. Currently only
        supports homoscedastic case.

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

    Attributes
    ----------
    results_: dict["PDS", "CHS"]
        Dictionary containing the 2-stage-least-squares estimates.
        Values are `linearmodels.iv.IV2SLS` objects. See:
        https://bashtage.github.io/linearmodels/iv/iv/linearmodels.iv.model.IV2SLS.html
        https://bashtage.github.io/linearmodels/iv/examples/basic-examples.html

    X_selected_: dict[list[str]]
        List of selected controls for each stage in the estimation.

    Z_selected_: list[str]
        List of selected instruments.

    valid_vars_: list[str]
        List of variables for which standard errors and test
        statistics are valid.

    References
    ----------
    Chernozhukov, V., Hansen, C., & Spindler, M. (2015).
        Post-selection and post-regularization inference in linear models with many controls and instruments.
        American Economic Review, 105(5), 486-90.

    Belloni, A., Chernozhukov, V., & Hansen, C. (2014).
        Inference on treatment effects after selection among high-dimensional controls.
        The Review of Economic Studies, 81(2), 608-650.

    Ahrens, A., Hansen, C. B., & Schaffer, M. (2019).
        PDSLASSO: Stata module for post-selection and
        post-regularization OLS or IV estimation and inference.
    """

    def __init__(
        self,
        *,
        select_X=True,
        select_Z=True,
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

        self.select_X = select_X
        self.select_Z = select_Z
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

    def _check_inputs(self, X, y, D_exog, D_endog, Z):
        """
        Checks inputs before passed to fit. For now, data is
        converted to pd.DataFrame's as it simplifices keeping track
        of nonzero indices and varnames significantly.
        """

        def _check_single(var, name):
            if var is None:
                return
            if isinstance(var, pd.DataFrame):
                return var

            if isinstance(var, np.ndarray):
                var = pd.DataFrame(var)
                var.columns = [f"{name}{i}" for i in range(var.shape[1])]
                return var

            elif isinstance(var, pd.core.series.Series):
                return (
                    pd.DataFrame(var)
                    if var.name
                    else pd.DataFrame(var, columns=[f"{name}"])
                )

            else:
                raise TypeError(
                    f"{name} must be a pandas dataframe or numpy array"
                    f"got {type(var)}"
                )

        X = _check_single(X, "X")
        y = _check_single(y, "y")
        D_exog = _check_single(D_exog, "d_exog")
        D_endog = _check_single(D_endog, "d_endog")
        Z = _check_single(Z, "Z")

        # save valid inference variables
        valid_vars = []
        if D_exog is not None:
            valid_vars += list(D_exog.columns.tolist())
        if D_endog is not None:
            valid_vars += list(D_endog.columns.tolist())

        return X, y, D_exog, D_endog, Z, valid_vars

    def _select_hd_vars(self, regressors, depvar):
        """
        Selects high-dimensional variables from the regressors.
        Returns nonzero idx, residuals and fitted values.
        """
        n, p = depvar.shape
        selected = []
        resid = np.empty((n, p))
        fitted = np.empty((n, p))

        for j in range(p):
            reg = self.rlasso.fit(regressors, depvar.iloc[:, j])
            [selected.append(i) for i in regressors.iloc[:, reg.nonzero_idx_].columns]
            pred = reg.predict(regressors)
            resid[:, j] = depvar.iloc[:, j] - pred
            fitted[:, j] = pred

        resid = pd.DataFrame(resid, columns=depvar.columns)
        fitted = pd.DataFrame(fitted, columns=depvar.columns)
        # return unique selected variables
        selected = list(set(selected))

        return selected, resid, fitted

    def _partial_ld_vars(self, regressors, depvar):

        n, p = depvar.shape
        resid = np.empty((n, p))
        fitted = np.empty((n, p))

        for j in range(p):
            tmp_dep = depvar.iloc[:, j].to_numpy()
            tmp_regressors = regressors.to_numpy()
            beta = self.rlasso._OLS(tmp_regressors, tmp_dep)
            pred = tmp_dep - tmp_regressors @ beta
            fitted[:, j] = pred
            resid[:, j] = depvar.iloc[:, j] - pred

        resid = pd.DataFrame(resid, columns=depvar.columns)
        fitted = pd.DataFrame(fitted, columns=depvar.columns)

        return resid, fitted

    def fit(
        self,
        X,
        y,
        D_exog=None,
        D_endog=None,
        Z=None,
    ):
        """
        Fit the model.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_controls)
            Control variables. Potentially high-dimensional.

        y: array-like, shape (n_samples,)
            Outcome/dependent variable.

        D_exog: array-like, shape (n_samples, n_exog)
            Low-dimensionnal exogenous regressors. On which inference
            is performed.

        D_endog: array-like, shape (n_samples, n_endog)
            Endogenous regressors. On which inference
            is performed.

        Z: array-like, shape (n_samples, n_instruments)
            Instruments. Potentially high-dimensional.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if self.select_Z and Z is None:
            raise ValueError("`select_Z=True` but no instruments `Z` provided")

        if Z is not None and D_endog is None:
            raise ValueError("Endogenous regressors D_endog must be provided")

        # check inputs
        X, y, D_exog, D_endog, Z, self.valid_vars_ = self._check_inputs(
            X, y, D_exog, D_endog, Z
        )

        if not self.select_X and X.shape[1] >= X.shape[0]:
            warnings.warn("`select_X=False` but X has more variables than observations")

        if not self.select_Z and Z.shape[1] >= Z.shape[0]:
            warnings.warn(
                "`select_Z=False` but Z has more instruments than observations"
            )

        # store all the selected variables
        X_selected = {}

        if self.select_X:
            # X_all_selected = []
            # step 1 (PDS/CHS). Select HD controls for dep var w.r.t. HD Xs
            X_selected["step_1"], rho_y, _ = self._select_hd_vars(X, y)

            # step 2 (PDS/CHS). Select HD controls for exog regressors w.r.t. HD Xs
            if D_exog is not None:
                X_selected["step_2"], rho_d, _ = self._select_hd_vars(X, D_exog)

            # step 3 (PDS). Select HD controls for endog regressors w.r.t. HD Xs
            if D_endog is not None:
                X_selected["step_3"], _, _ = self._select_hd_vars(X, D_endog)

            # store all the selected X's
            self.X_selected_ = X_selected

        # handle CHS residuals in the case of
        # X's not being penalized `select_X=False`
        else:
            rho_y, _ = self._partial_ld_vars(X, y)

            if D_exog is not None:
                rho_d, _ = self._partial_ld_vars(X, D_exog)

        if self.select_Z:
            # step 5 (PDS/CHS). Select HD controls for Z w.r.t. HD Xs
            Z_selected, _, d_hat = self._select_hd_vars(
                pd.concat([Z, X], axis=1), D_endog
            )
            Z_selected = [s for s in Z_selected if s in Z.columns]
            Z = Z.loc[:, Z_selected]
            # elif D_endog is not None:
            #     self.X_selected_4_ = self._select_hd_vars(X, Z)
            #     X_all_selected += self.X_selected_4_

            # step 6 (CHS). Create optimal instrument for endog
            X_selected["step_6"], iv_e, _ = self._select_hd_vars(X, d_hat)
            # iv_e.columns = Z_selected

            # store all the selected instruments
            self.Z_selected_ = Z_selected

            # step 7 (CHS). Create orthogonalized endog
            rho_e = pd.DataFrame(
                D_endog.to_numpy() - (d_hat.to_numpy() - iv_e),
                columns=D_endog.columns,
            )
        # handle CHS residuals in the case of Z's not being
        # penalized `select_Z=False`
        elif D_endog is not None:
            _, d_hat = self._partial_ld_vars(pd.concat([Z, X], axis=1), D_endog)

            iv_e, _ = self._partial_ld_vars(X, d_hat)

            # step 7 (CHS). Create orthogonalized endog
            rho_e = pd.DataFrame(
                D_endog.to_numpy() - (d_hat.to_numpy() - iv_e),
                columns=D_endog.columns,
            )

        # adjust for variation in naming for homoscedastic case
        cov_type = "unadjusted" if self.cov_type == "nonrobust" else self.cov_type

        # fit CHS IV2SLS
        chs = IV2SLS(
            rho_y,
            rho_d if "rho_d" in locals() else None,
            rho_e if "rho_e" in locals() else None,
            iv_e.to_numpy() if "iv_e" in locals() else None,
        ).fit(cov_type=cov_type)

        # fit PDS IV2SLS
        # get unique X selected
        if self.select_X:
            X_unique_mask = []
            for step, var in X_selected.items():
                # only for chs
                if step != "step_6":
                    X_unique_mask += var

            X_unique_mask = list(set(X_unique_mask))
            if not X_unique_mask:
                warnings.warn("No controls in X where selected")
            X = X.loc[:, X_unique_mask]

        if D_exog is not None:
            X = pd.concat([D_exog, X], axis=1)

        if self.fit_intercept:
            X = add_constant(X)

        else:
            X = D_exog if D_exog is not None else None

        pds = IV2SLS(
            y,
            X,
            D_endog if D_endog is not None else None,
            Z if Z is not None else None,
        ).fit(cov_type=cov_type)

        self.results_ = {"CHS": chs, "PDS": pds}

        return self

    def summary(self):
        """
        Produces a summary of the results.

        """

        check_is_fitted(self, "results_")

        return compare(
            {
                "PDS": self.results_["PDS"],
                "CHS": self.results_["CHS"],
            },
            stars=True,
            precision="std_errors",
        )


class RlassoPDS(RlassoIV):
    """
    Rigorous Lasso for causal inference of low-dimensional
    exogenous regressors in the presence of high-dimensional
    controls. Uses the post-double-selection (PDS) and
    post-regularization (CHS) methods for estimation, see
    references below.

    Parameters
    ----------
    post: bool, default=True
        If True, post-lasso is used to estimate betas.
        Note that `post` will only affect the results
        for the post-regularization (CHS) method
        and not those of post-double-selection (pds).

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

    prestd: bool, default=False
        If True, the data is prestandardized instead of
        on the fly by penalty loadings. Currently only
        supports homoscedastic case.

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

    Attributes
    ----------
    results_: dict["PDS", "CHS"]
        Dictionary containing the 2-stage-least-squares estimates.
        Values are `linearmodels.iv.IV2SLS` objects. See:
        https://bashtage.github.io/linearmodels/iv/iv/linearmodels.iv.model.IV2SLS.html
        https://bashtage.github.io/linearmodels/iv/examples/basic-examples.html

    X_selected_: dict[list[str]]
        List of selected controls for each stage in the estimation.

    valid_vars_: list[str]
        List of variables for which standard errors and test
        statistics are valid.

    References
    ----------
    Chernozhukov, V., Hansen, C., & Spindler, M. (2015).
        Post-selection and post-regularization inference in linear models with many
        controls and instruments. American Economic Review, 105(5), 486-90.

    Belloni, A., Chernozhukov, V., & Hansen, C. (2014).
        Inference on treatment effects after selection among high-dimensional controls.
        The Review of Economic Studies, 81(2), 608-650.

    Ahrens, A., Hansen, C. B., & Schaffer, M. (2019).
        PDSLASSO: Stata module for post-selection and
        post-regularization OLS or IV estimation and inference.
    """

    def __init__(
        self,
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

        super().__init__(
            select_X=True,
            select_Z=False,
            post=post,
            sqrt=sqrt,
            fit_intercept=fit_intercept,
            cov_type=cov_type,
            x_dependent=x_dependent,
            random_state=random_state,
            lasso_psi=lasso_psi,
            prestd=prestd,
            n_corr=n_corr,
            max_iter=max_iter,
            conv_tol=conv_tol,
            n_sim=n_sim,
            c=c,
            gamma=gamma,
            solver=solver,
            cd_max_iter=cd_max_iter,
            cd_tol=cd_tol,
            cvxpy_opts=cvxpy_opts,
            zero_tol=zero_tol,
        )

    def fit(self, X, y, D_exog):
        """
        Fit the model.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_controls)
            High-dimensional control variables.

        y: array-like, shape (n_samples,)
            Outcome/dependent variable.

        D_exog: array-like, shape (n_samples, n_exog)
            Low-dimensionnal exogenous regressors. On which inference
            is performed.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        return super().fit(X, y, D_exog)
