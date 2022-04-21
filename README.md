[![Conda Actions Status][actions-conda-badge]][actions-conda-link] 
[![Pip Actions Status][actions-pip-badge]][actions-pip-link] 
[![Docs Actions Status][actions-docs-badge]][actions-docs-link]
[![Python](https://img.shields.io/badge/python-3.6%20%7C%203.8%20%7C%203.10-blue)](https://www.python.org)

<!-- [![**Docs**][docs-link] -->


[actions-badge]:           https://github.com/matpiq/rlassopy/workflows/Tests/badge.svg
[actions-conda-link]:      https://github.com/matpiq/rlassopy/actions?query=workflow%3AConda
[actions-conda-badge]:     https://github.com/matpiq/rlassopy/workflows/Conda/badge.svg
[actions-pip-link]:        https://github.com/matpiq/rlassopy/actions?query=workflow%3APip
[actions-pip-badge]:       https://github.com/matpiq/rlassopy/workflows/Pip/badge.svg
[actions-wheels-link]:     https://github.com/matpiq/rlassopy/actions?query=workflow%3AWheels
[actions-wheels-badge]:    https://github.com/matpiq/rlassopy/workflows/Wheels/badge.svg
[actions-docs-link]:       https://rlassopy.readthedocs.io/en/latest/?badge=latest
[actions-docs-badge]:      https://readthedocs.org/projects/rlassopy/badge/?version=latest



# rlassopy

[rlassopy]: https://rlasso.readthedocs.io/en/latest/
[lassopack]: https://statalasso.github.io/docs/lassopack/
[hdm]: https://CRAN.R-project.org/package=hdm
[documentation]: https://rlasso.readthedocs.io/en/latest/user_guide.html


The package implements rigorous Lasso for estimation and inference in high-dimensional datasets. 
It aims to provide a Python alternative to the existing stata packages [`lassopack`](https://statalasso.github.io/docs/lassopack/) and
[pdslasso](https://statalasso.github.io/docs/pdslasso/) (Ahrens, Hansen & Schaffer, 2018, 2020) and the R package 
[hdm](https://CRAN.R-project.org/package=hdm). 

# Main features

The class `Rlasso` provides 


## References

Ahrens A, Hansen CB, Schaffer ME (2020). lassopack: Model selection and prediction with regularized regression in Stata. The Stata Journal. 20(1):176-235. doi:10.1177/1536867X20909697

Ahrens, A., Hansen, C.B., Schaffer, M.E. 2018. pdslasso and ivlasso: Programs for post-selection and post-regularization OLS or IV estimation and inference. http://ideas.repec.org/c/boc/bocode/s458459.html

Chernozhukov, V., Hansen, C., & Spindler, M. (2016). hdm: High-dimensional metrics. arXiv preprint arXiv:1608.00354.
