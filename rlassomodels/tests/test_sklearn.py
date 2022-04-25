import pytest
from sklearn.utils.estimator_checks import check_estimator

from rlassomodels import Rlasso


@pytest.mark.parametrize("estimator", [Rlasso()])
def test_all_estimators(estimator):
    return check_estimator(estimator)
