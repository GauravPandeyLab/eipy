Welcome to Ensemble Integration's documentation!
================================================

**Ensemble Integration (eipy)** leverages multi-modal data to build classifiers using a late fusion approach. 
In eipy, base predictors are trained on each modality before being ensembled at the late stage. 

This implementation of eipy can utilize `sklearn-like <https://scikit-learn.org/>`_ models only, therefore, for unstructured data,
e.g. images, it is recommended to perform feature selection prior to using eipy. We hope to allow for a wider range of base predictors, 
i.e. deep learning methods, in future releases.

For more details see the `original publication <https://doi.org/10.1093/bioadv/vbac065>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   getting_started
   tutorial.ipynb
   api_reference
   development

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
