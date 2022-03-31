import pytest
from sklearn.utils.estimator_checks import check_estimator

from rlassopy import Rlasso, RlassoLogit


@pytest.mark.parametrize("estimator", [Rlasso(), RlassoLogit()])
def test_all_estimators(estimator):
    return check_estimator(estimator)
