Getting started
===============

The purpose of Ensemble Integration (EI) is to fuse multi-modal data using late fusion. Simply concatenating features is not
always guaranteed to generate the best performance. In EI base predictors can be trained on each modality before
being ensembled at the late stage. 

This implementation of EI utilises `sklearn <https://scikit-learn.org/>`_ models only, therefore, for unstructured data 
it is recommended to perform feature selection prior to using EI.

Source code
-----------

The source code for Ensemble Integration is available on `github <https://github.com/03bennej/ei-python.git>`_.

Installation
------------

As usual it is recommended to set up a virtual environment prior to installation. 
To use Ensemble Integration, first install it using pip:

.. code-block:: console

   pip install "git+https://github.com/03bennej/ei-python.git"

