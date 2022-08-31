# Ensemble Integration (EI): Integrating multimodal data through interpretable heterogeneous ensembles
Ensemble Integration (EI) is a customizable pipeline for generating diverse ensembles of heterogeneous classifiers, as well as the accompanying metadata needed for ensemble learning approaches utilizing ensemble diversity for improved performance. It also fairly evaluates the performance of several ensemble learning methods including ensemble selection [Caruana2004], and stacked generalization (stacking) [Wolpert1992]. Though other tools exist, we are unaware of a similar modular, scalable pipeline designed for large-scale ensemble learning. This fully python version of EI was implemented by Jamie J. R. Bennett and Yan Chak Li, and is based on the original version: https://github.com/GauravPandeyLab/ensemble_integration.git.

EI is designed for generating extremely large ensembles (taking days or weeks to generate) and thus consists of an initial data generation phase tuned for multicore and distributed computing environments. The output is a set of compressed CSV files containing the class distribution produced by each classifier that serves as input to a later ensemble learning phase.

More details of EI can be found in our Biorxiv preprint:

Full citation:

Yan Chak Li, Linhua Wang, Jeffrey Law, T. M. Murali, Gaurav Pandey (2020): Integrating multimodal data through interpretable heterogeneous ensembles, bioRxiv. Preprint. 2020.05.29.123497; doi: https://doi.org/10.1101/2020.05.29.123497

This repository is protected by CC-BY-NC-ND-4.0.

## Requirements ##

`
python==3.9.12

scikit-learn==1.1.1

pandas==1.4.3

numpy==1.22.3

joblib==1.1.0

`
