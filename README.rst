|Tests|_ |ReadTheDocs|_ |PythonVersion|_ |Black|_ |License|_

.. |Tests| image:: https://github.com/GauravPandeyLab/eipy/actions/workflows/tests.yml/badge.svg
.. _Tests: https://github.com/GauravPandeyLab/eipy/actions/workflows/tests.yml

.. |ReadTheDocs| image:: https://readthedocs.org/projects/eipy/badge/?version=latest
.. _ReadTheDocs: https://eipy.readthedocs.io/en/latest/

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue
.. _PythonVersion: https://github.com/GauravPandeyLab/eipy

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _Black: https://github.com/psf/black

.. |License| image:: https://img.shields.io/badge/License-GPLv3-blue
.. _License: https://github.com/GauravPandeyLab/eipy/blob/main/COPYING


``ensemble-integration``: Integrating multi-modal data for predictive modeling
==============================================================================

``ensemble-integration`` (or ``eipy``) leverages multi-modal data to build classifiers using a late fusion approach. 
In eipy, base predictors are trained on each modality before being ensembled at the late stage. 

This implementation of eipy can utilize `sklearn-like <https://scikit-learn.org/>`_ models only, therefore, for unstructured data,
e.g. images, it is recommended to perform feature selection prior to using eipy. We hope to allow for a wider range of base predictors, 
i.e. deep learning methods, in future releases. A key feature of ``eipy`` is its built-in nested cross-validation approach, allowing for a 
fair comparison of a collection of user-defined ensemble methods.

Documentation including tutorials are available at `https://eipy.readthedocs.io/en/latest/ <https://eipy.readthedocs.io/en/latest/>`_.

Installation
------------

As usual it is recommended to set up a virtual environment prior to installation. 
You can install ensemble-integration with pip:

``pip install ensemble-integration``

Citation
--------

If you use ``ensemble-integration`` in a scientific publication please cite the following:

Jamie J. R. Bennett, Yan Chak Li and Gaurav Pandey. *An Open-Source Python Package for Multi-modal Data Integration using Heterogeneous Ensembles*, https://doi.org/10.48550/arXiv.2401.09582.

Yan Chak Li, Linhua Wang, Jeffrey N Law, T M Murali, Gaurav Pandey. *Integrating multimodal data through interpretable heterogeneous ensembles*, Bioinformatics Advances, Volume 2, Issue 1, 2022, vbac065, https://doi.org/10.1093/bioadv/vbac065.

