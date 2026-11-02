pydaarem
========

pydaarem is a Python port of the R package
`daarem <https://CRAN.R-project.org/package=daarem>`_. It accelerates smooth,
monotone fixed-point iterations, such as the expectation maximization (EM)
algorithm, with damped Anderson acceleration with restarts and
epsilon-monotonicity (DAAREM).

The public API and the control names are the same as in the R package. Tests
compare the results of pydaarem with the results of the R package.

Install
-------

pydaarem is not on PyPI. Install it from a clone of the repository:

.. code-block:: bash

   pip install .

To use the example data sets, install the ``datasets`` extra:

.. code-block:: bash

   pip install ".[datasets]"

Reference
---------

Henderson, N. C. and Varadhan, R. (2019). Damped Anderson acceleration with
restarts and monotonicity control for accelerating EM and EM-like algorithms.
*Journal of Computational and Graphical Statistics*, 28(4), 834-846.
https://doi.org/10.1080/10618600.2019.1594835

.. toctree::
   :hidden:

   api
   auto_examples/index
