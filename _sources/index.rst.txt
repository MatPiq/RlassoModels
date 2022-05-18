.. RlassoModels documentation master file, created by
   sphinx-quickstart on Sun Apr 24 17:26:10 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to RlassoModels!
========================================

.. _rlassopy: https://rlassomodels.readthedocs.io/en/latest/
.. _lassopack: https://statalasso.github.io/docs/lassopack/
.. _pdslasso: https://statalasso.github.io/docs/pdslasso/
.. _hdm: https://CRAN.R-project.org/package=hdm
.. _documentation: https://rlassomodels.readthedocs.io/en/latest/user_guide.html

RlassoModels implements rigorous Lasso for estimation and inference in high-dimensional datasets. 
It aims to provide a Python alternative to the existing stata packages lassopack_ and pdslasso_ (Ahrens, Hansen & Schaffer, 2018, 2020) and the R package hdm_ (Chernozkukov, Hansen & Spindler, 2016).

.. code:: python
    
    from sklearn.datasets import make_regression
    from rlassomodels import Rlasso

    # Generate data.
    X, y = make_regression(n_samples=100, n_features=80, n_informative=5)

    # Define the model and fit to data
    model = Rlasso()
    model.fit(X, y)

.. toctree::
   :hidden:

   install

.. toctree::
   :hidden:

   examples

.. toctree::
   :hidden:

   api_reference
