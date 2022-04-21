[![Conda Actions Status][actions-conda-badge]][actions-conda-link] 
[![Pip Actions Status][actions-pip-badge]][actions-pip-link] 
[![Docs Actions Status][actions-docs-badge]][actions-docs-link]
[![Python](https://img.shields.io/badge/python-3.6%20%7C%203.8%20%7C%203.10-blue)](https://www.python.org)

<!-- [![**Docs**][docs-link] -->


[actions-badge]:           https://github.com/matpiq/RlassoModels/workflows/Tests/badge.svg
[actions-conda-link]:      https://github.com/matpiq/RlassoModels/actions?query=workflow%3AConda
[actions-conda-badge]:     https://github.com/matpiq/RlassoModels/workflows/Conda/badge.svg
[actions-pip-link]:        https://github.com/matpiq/RlassoModels/actions?query=workflow%3APip
[actions-pip-badge]:       https://github.com/matpiq/RlassoModels/workflows/Pip/badge.svg
[actions-wheels-link]:     https://github.com/matpiq/RlassoModels/actions?query=workflow%3AWheels
[actions-wheels-badge]:    https://github.com/matpiq/RlassoModels/workflows/Wheels/badge.svg
[actions-docs-link]:       https://RlassoModels.readthedocs.io/en/latest/?badge=latest
[actions-docs-badge]:      https://readthedocs.org/projects/RlassoModels/badge/?version=latest



# RlassoModels - Rigorous Lasso for Estimation and Inference in Python

[RlassoModels]: https://RlassoModels.readthedocs.io/en/latest/
[lassopack]: https://statalasso.github.io/docs/lassopack/
[hdm]: https://CRAN.R-project.org/package=hdm


The package implements rigorous Lasso for estimation and inference in high-dimensional datasets. 
It aims to provide a Python alternative to the existing stata packages [lassopack](https://statalasso.github.io/docs/lassopack/) and
[pdslasso](https://statalasso.github.io/docs/pdslasso/) (Ahrens, Hansen & Schaffer, 2018, 2020) and the R package 
[HDM](https://CRAN.R-project.org/package=hdm) (Chernozkukov, Hansen & Spindler, 2016). Documentation is available at [RlassoModels](https://RlassoModels.readthedocs.io/en/latest/). For a general introduction to the methods, I suggest the accompanying papers to lassopack and HDM.

## Main features

The class `Rlasso` is a [scikit-learn](https://scikit-learn.org/stable/) compatible estimator that implements the lasso
and square-root lasso with data-driven and theoretically justified penalty level. `RlassoPDS` and `RlassoIV` extend
`Rlasso` and implement post-double-selection and post-regularization/partialling-out inference on low-dimensional variables in the
presence of high-dimensional controls and/or instruments. The theory was developed in series of papers 
by Belloni et al. (2011, 2013, 2014) and Chernozhukov et al. (2015).

## Installation

`RlassoModels` depends on:

* [`numpy`](https://numpy.org/)
* [`scipy`](https://www.scipy.org/)
* [`scikit-learn`](https://scikit-learn.org/)
* [`pandas`](https://pandas.pydata.org/)
* [`cvxpy`](https://www.cvxpy.org/)
* [`patsy`](https://patsy.readthedocs.io/en/latest/)
* [`linearmodels`](https://bashtage.github.io/linearmodels)

It can be installed be installed from source by

```
git clone git@github.com:matpiq/RlassoModels.git
cd RlassoModels
pip install .
```
Note that the package is still experimental and in development. It is therefore not yet available
on PyPI and Conda-Forge.

## References

Ahrens A, Hansen CB, Schaffer ME (2020). lassopack: Model selection and prediction with regularized regression in Stata. The Stata Journal. 20(1):176-235. doi:10.1177/1536867X20909697

Ahrens, A., Hansen, C.B., Schaffer, M.E. 2018. pdslasso and ivlasso: Programs for post-selection and post-regularization OLS or IV estimation and inference. http://ideas.repec.org/c/boc/bocode/s458459.html

Chernozhukov, V., Hansen, C., & Spindler, M. (2016). hdm: High-dimensional metrics. arXiv preprint arXiv:1608.00354.

Belloni, Alexandre, Victor Chernozhukov, and Lie Wang. "Square-root lasso: pivotal recovery of sparse signals via conic programming." Biometrika 98.4 (2011): 791-806.

Belloni, A., & Chernozhukov, V. (2013). Least squares after model selection in high-dimensional sparse models. Bernoulli, 19(2), 521-547.

Belloni, A., Chernozhukov, V., & Hansen, C. (2014). High-dimensional methods and inference on structural and treatment effects. Journal of Economic Perspectives, 28(2), 29-50.

Chernozhukov, V., Hansen, C., & Spindler, M. (2015). Post-selection and post-regularization inference in linear models with many controls and instruments. American Economic Review, 105(5), 486-90.
