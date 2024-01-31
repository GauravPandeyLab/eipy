Getting started
===============

Ensemble Integration focuses mainly on
`stacked generalization <https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231>`_,
as a method for late data fusion, but other ensemble methods including 
`ensemble selection <https://dl.acm.org/doi/10.1145/1015330.1015432>`_ are available for
comparison. 

Base predictor training is performed in a nested cross validation set up, to allow for an unbiased comparison
of ensemble methods, allowing the user to select the method with the best performance. 
A final model can then be trained on all available data.

Source code
-----------

The source code for eipy is available on `GitHub <https://github.com/03bennej/ei-python.git>`_.

Installation
------------

As usual it is recommended to set up a virtual environment prior to installation. 
You can install ensemble-integration with pip:

.. code-block:: console

   pip install ensemble-integration

Citation
--------

If you use eipy in a scientific publication please cite the following:

Jamie J. R. Bennett, Yan Chak Li and Gaurav Pandey. An Open-Source Python Package for Multi-modal Data Integration using Heterogeneous Ensembles, https://doi.org/10.48550/arXiv.2401.09582.

Yan Chak Li, Linhua Wang, Jeffrey N Law, T M Murali, Gaurav Pandey. Integrating multimodal data through interpretable heterogeneous ensembles, Bioinformatics Advances, Volume 2, Issue 1, 2022, vbac065, https://doi.org/10.1093/bioadv/vbac065.

