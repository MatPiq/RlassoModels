import numpy as np
import numpy.linalg as la
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_equal

from rlassomodels import Rlasso


@pytest.fixture
def data_sparse():
    """
    Data-generating function following Belloni (2011).
    """
    np.random.seed(234923)

    # Based on the example in the Belloni paper
    n = 100
    p = 500
    ii = np.arange(p)
    cx = 0.5 ** np.abs(np.subtract.outer(ii, ii))
    cxr = np.linalg.cholesky(cx)

    X = np.dot(np.random.normal(size=(n, p)), cxr.T)
    b = np.zeros(p)
    b[:5] = [1, 1, 1, 1, 1]
    y = np.dot(X, b) + 0.25 * np.random.normal(size=n)

    return X, y, b, cx


@pytest.fixture
def data_dummy():

    np.random.seed(234923)
    n = 100
    p = 5
    X = np.random.normal(size=(n, p))
    b = np.ones(p)
    y = X @ b

    return X, y


def test_post(data_dummy):
    """
    Test that post estimation yields expected OLS results.
    """
    X, y = data_dummy

    # normal rlasso
    post = Rlasso(post=True).fit(X, y)
    sqrt_post = Rlasso(sqrt=True, post=True).fit(X, y)
    ols = la.solve(X.T @ X, X.T @ y)

    assert_allclose(post.coef_, ols, rtol=1e-3, atol=1e-3)
    assert_allclose(sqrt_post.coef_, ols, rtol=1e-3, atol=1e-3)


def test_prestd(data_dummy):
    """Test that the pre-standardization yield
    the same result as "on-the-fly" standardization.
    through penalty loadings.
    """
    X, y = data_dummy
    beta_psi = Rlasso(post=False, prestd=False).fit(X, y).coef_
    beta_prestd = Rlasso(post=False, prestd=True).fit(X, y).coef_

    assert_allclose(beta_psi, beta_prestd, rtol=1e-5, atol=1e-5)


def test_formula(data_dummy):
    """
    Test that the formula parser works.
    """
    X, y = data_dummy
    df = pd.DataFrame(np.c_[y, X])

    df.columns = ["y"] + [f"x{i}" for i in range(X.shape[1])]
    formula = "y ~ " + " + ".join(df.columns.tolist()[1:])

    res1 = Rlasso().fit(X, y)
    res2 = Rlasso().fit_formula(formula, data=df)

    assert_equal(res1.coef_, res2.coef_)


def test_cd_vs_cvxpy(data_sparse):
    """
    Test that the CD and CVXPY solvers give the same results.
    """
    X, y, _, _ = data_sparse

    # rlasso
    res_cvxpy = Rlasso(post=False, solver="cvxpy").fit(X, y)
    res_cd = Rlasso(post=False, solver="cd").fit(X, y)

    assert_allclose(res_cvxpy.coef_, res_cd.coef_, atol=1e-5)

    # sqrt-rlasso
    res_sqrt_cvxpy = Rlasso(sqrt=True, post=False, solver="cvxpy").fit(X, y)
    res_sqrt_cd = Rlasso(sqrt=True, post=False, solver="cd").fit(X, y)

    assert_allclose(res_sqrt_cvxpy.coef_, res_sqrt_cd.coef_, atol=1e-5)


def test_rlasso_oracle(data_sparse):
    """
    Same test as `statsmodels.regression.tests_regression.test_sqrt_lasso`
    with addition of test for selected components.
    Based on SQUARE-ROOT LASSO: PIVOTAL RECOVERY OF SPARSE
    SIGNALS VIA CONIC PROGRAMMING, Belloni (2011), p.10.
    """
    X, y, b, cx = data_sparse
    _, p = X.shape

    # Empirical risk ratio.
    expected_oracle = {False: 3, True: 1}

    # Used for regression testing
    expected_params = {
        # False: np.r_[0.86706825, 1.00475367, 0.98628392, 0.93160201, 0.9293992],
        False: np.r_[0.86709638, 1.00424588, 0.98749245, 0.93101511, 0.92977734],
        True: np.r_[0.95300153, 1.03060962, 1.01297103, 0.97404348, 1.00306961],
    }

    for post in False, True:

        res = Rlasso(sqrt=False, post=post).fit(X, y)
        e = res.coef_ - b
        numer = np.sqrt(np.dot(e, np.dot(cx, e)))

        X_oracle = X[:, :5]
        oracle = la.inv(X_oracle.T @ X_oracle) @ X_oracle.T @ y
        oracle_e = np.zeros(p)
        oracle_e[:5] = oracle - b[:5]
        denom = np.sqrt(np.dot(oracle_e, np.dot(cx, oracle_e)))

        # Check performance relative to oracle, should be around 3.5 for
        # post=False, 1 for post=True
        assert_allclose(numer / denom, expected_oracle[post], rtol=0.5, atol=0.1)

        # Check number of selected components relative to oracle,
        # should be equal for small noise lvls
        n_components = np.nonzero(res.coef_)[0].size
        assert n_components == 5

        # Regression test the parameters
        assert_allclose(res.coef_[:5], expected_params[post], rtol=1e-5, atol=1e-5)


def test_sqrt_rlasso_oracle(data_sparse):
    """
    Same as `test_rlasso_oracle` but with sqrt=True.
    Note empirical risk ratio is 3.5 and
    different from `test_rlasso_oracle`.
    """
    X, y, b, cx = data_sparse
    _, p = X.shape

    # Empirical risk ratio. Note: statsmodels uses
    # 3.0, this should be 3.5 (see the paper)
    expected_oracle = {False: 3.5, True: 1}

    # Used for regression testing
    expected_params = {
        # False: np.r_[0.83455166, 0.99496994, 0.97618569, 0.91554244, 0.9015228],
        False: np.r_[0.83481394, 0.9943386, 0.97791781, 0.91484541, 0.90223471],
        True: np.r_[0.95300153, 1.03060962, 1.01297103, 0.97404348, 1.00306961],
    }

    for post in False, True:

        res = Rlasso(sqrt=True, post=post).fit(X, y)
        e = res.coef_ - b
        numer = np.sqrt(np.dot(e, np.dot(cx, e)))

        X_oracle = X[:, :5]
        oracle = la.inv(X_oracle.T @ X_oracle) @ X_oracle.T @ y
        oracle_e = np.zeros(p)
        oracle_e[:5] = oracle - b[:5]
        denom = np.sqrt(np.dot(oracle_e, np.dot(cx, oracle_e)))

        # Check performance relative to oracle, should be around 3.5 for
        # post=False, 1 for post=True
        assert_allclose(numer / denom, expected_oracle[post], rtol=0.5, atol=0.1)

        # Check number of selected components relative to oracle,
        # should be equal for small noise lvls
        n_components = np.nonzero(res.coef_)[0].size
        assert n_components == 5

        # Regression test the parameters
        assert_allclose(res.coef_[:5], expected_params[post], rtol=1e-5, atol=1e-5)


def test_rlasso_vs_lassopack(data_sparse):
    """
    Test that rlasso and lassopack implementation are equivalent
    on Belloni data. Stata specifications:
        . rlasso y x1-x500
        . rlasso y x1-x500, robust
        . rlasso y x1-x500, sqrt
        . rlasso y x1-x500, sqrt, robust
    """

    comparison_tab = {
        "rlasso": {
            "model": Rlasso(post=False),
            "lp_coef": [0.8670964, 1.0042459, 0.9874925, 0.9310151, 0.9297773],
            "lp_lambd": 23.002404,
        },
        "rlasso_robust": {
            "model": Rlasso(post=False, cov_type="robust"),
            "lp_coef": [0.8782264, 1.0137120, 0.9512580, 0.9571405, 0.9249149],
            "lp_lambd": 89.945555,
        },
        "sqrt_rlasso": {
            "model": Rlasso(sqrt=True, post=False),
            "lp_coef": [0.8348139, 0.9943386, 0.9779178, 0.9148454, 0.9022347],
            "lp_lambd": 44.972777,
        },
        "sqrt_rlasso_robust": {
            "model": Rlasso(sqrt=True, post=False, cov_type="robust"),
            "lp_coef": [0.8347708, 1.0104739, 0.9380134, 0.9295666, 0.9013439],
            "lp_lambd": 44.972777,
        },
        "post_rlasso": {
            "model": Rlasso(post=True),
            "lp_coef": [0.9530015, 1.0306096, 1.0129710, 0.9740435, 1.0030696],
        },
    }

    X, y, _, _ = data_sparse

    for m_name, spec in comparison_tab.items():

        # Compare the model
        model = spec["model"].fit(X, y)
        coef = model.coef_[np.nonzero(model.coef_)]
        lambd = model.lambd_

        # compare coefs
        # low precision due to sqrt-lasso. Unknown why
        assert_allclose(coef, spec["lp_coef"], atol=1e-5, rtol=1e-5)

        # compare lambd
        if m_name != "post_rlasso":
            assert_allclose(lambd, spec["lp_lambd"], atol=1e-5)
