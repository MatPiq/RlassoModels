import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.datasets import load_iris

from rlassopy import Rlasso, RlassoLogit


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_template_estimator(data):
    est = Rlasso()
    assert est.cov_type == "nonrobust"

    est.fit(*data)
    assert hasattr(est, "is_fitted_")

    X = data[0]
    y_pred = est.predict(X)
    assert_array_equal(y_pred, np.ones(X.shape[0], dtype=np.int64))


# def test_template_classifier(data):
# TODO test classifier
# X, y = data
# clf = TemplateClassifier()
# assert clf.demo_param == "demo"
#
# clf.fit(X, y)
# assert hasattr(clf, "classes_")
# assert hasattr(clf, "X_")
# assert hasattr(clf, "y_")
#
# y_pred = clf.predict(X)
# assert y_pred.shape == (X.shape[0],)
