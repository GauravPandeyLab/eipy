from setuptools import setup, find_packages

setup(name='eipy', 
      version='0.0.1', 
      description='Ensemble Integration (EI)',
      url='https://github.com/03bennej/ei-python.git',
      author='Jamie Bennett, Richard Li',
      license='CC by 4.0',
      packages=find_packages(),
      install_requires=["imbalanced_learn",
                        "joblib",
                        "numpy",
                        "pandas",
                        "scikit_learn",
                        "scipy",
                        "setuptools",
                        "shap",
                        "xgboost"]
      )