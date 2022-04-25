import numpy as np
import numpy.linalg as la
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_equal

from rlassomodels import RlassoIV, RlassoPDS


@pytest.fixture
def ajr_data():
    return pd.read_stata("https://statalasso.github.io/dta/AJR.dta")


def extract_params(model):
    pds = model.results_["PDS"]
    chs = model.results_["CHS"]
    return {
        "pds_beta": pds.params,
        "chs_beta": chs.params,
    }
