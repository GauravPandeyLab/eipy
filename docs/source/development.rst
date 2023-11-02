Development
===========

We welcome contributions to the development of ``eipy``. To contribute follow the below instructions to submit a pull request:

1. **Install Python**. First of all make sure you have a supported version of Python on your local machine (see `GitHub <https://github.com/GauravPandeyLab/eipy>`__ for supported versions).
2. **Install Poetry**. ``eipy`` uses Poetry to manage dependencies. To install Poetry follow the instructions on their `website <https://python-poetry.org/docs/>`__.
3. **Fork the repo**.

.. code-block:: console

   git fork https://github.com/GauravPandeyLab/eipy.git

4. **Set up a virtual environment**. Navigate to the ``eipy`` directory and create and activate a virtual environment.

.. code-block:: console

   python -m venv .venv
   source .venv/bin/activate

5. **Install dependencies**. If editing the documentation make sure to include the ``--with docs`` argument.

.. code-block:: console

   poetry install --with docs

6. **Make contributions**.

7. **Linting and formating**. We use Flake8 for linting and Black for formatting. For linting type, for example,

.. code-block:: console

   flake8 eipy/ei

For formatting type, for example,

.. code-block:: console

   black eipy/ei

8. **Run tests**. All tests can be found in the tests folder and can be run by typing

.. code-block:: console

   pytest

Note that new test file names must have the prefix `test_`.

9. **Submit pull request**. Updates must be made via a pull request. Internal users should note that pushing to the main branch has been disabled.

10. **Publishing new versions to PyPI** (internal only). To release new versions of ``eipy`` to PyPI make sure you have a PyPI account
and have been added as a collaborater to the project. When you want to published, change the version number in the ``[tool.poetry]`` section
of the `pyproject.toml` and push changes to a secondary branch, before merging with the main branch. Switch to the main branch and sync changes.
You can now type:

.. code-block:: console

   poetry publish

Note that versions cannot be deleted, and bug fixes etc must be published as a new version. After this has been done publish a new release on GitHub
making sure to match the version number with that in `pyproject.toml`.
