Getting started
===============

eipy focuses mainly on
`stacked generalization <https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231>`_,
as a method for late data fusion, but other ensemble methods including 
`ensemble selection <https://dl.acm.org/doi/10.1145/1015330.1015432>`_ are available for
comparison. 

Base predictor training is performed in a nested cross validation set up, to allow for an unbiased comparison
and selection the ensembmle method with the best performance. A final model can then be trained on all available
data.

Source code
-----------

The source code for eipy is available on `GitHub <https://github.com/03bennej/ei-python.git>`_.

Installation
------------

As usual it is recommended to set up a virtual environment prior to installation. 
To use eipy, first install it using pip:

.. code-block:: console

   pip install "git+https://github.com/03bennej/ei-python.git"

Citation
--------

If you use eipy in a scientific publication please cite the `original paper <https://doi.org/10.1093/bioadv/vbac065>`_.

