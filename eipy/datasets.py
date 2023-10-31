import pandas as pd
import os
from os import environ, makedirs
from os.path import expanduser, join
import wget
import zipfile


def load_diabetes():
    """
    Loads a multi-modal youth diabetes dataset.

    More information about this dataset can be found in the following publication:

    Catherine McDonough, Yan Chak Li, Nita Vangeepuram, Bian Liu, Gaurav Pandey.
    Facilitating youth diabetes studies with the most comprehensive epidemiological
    dataset available through a public web portal. medRxiv 2023.08.02.23293517.
    https://doi.org/10.1101/2023.08.02.23293517

    Returns
    -------
    data : dict
        Dictionary with keys 'X_train', 'y_train', 'X_test', 'y_test', 'data_dict'.

    """
    zenodo_link = "https://zenodo.org/records/10035422/files/diabetes.zip?download=1"
    # Get data path
    data_path = get_data_home()
    folder_ext = "diabetes"
    data_ext_path = join(data_path, folder_ext)
    # check data downloaded before
    folder_exist = os.path.exists(data_ext_path)
    zip_exist = os.path.exists(data_ext_path + ".zip")
    if not folder_exist:
        if not zip_exist:
            wget.download(zenodo_link, out=data_path)
        downloaded_path = data_ext_path + ".zip"
        with zipfile.ZipFile(downloaded_path, "r") as zip_ref:
            zip_ref.extractall(data_path)

    _file_path = data_ext_path
    modality_keys = [
        "Sociodemographic",
        "Health status",
        "Diet",
        "Other lifestyle behaviors",
    ]
    _train_suffix = "9916"
    _test_suffix = "1618"
    X_train = {k: _load_csv(_file_path, k, _train_suffix) for k in modality_keys}
    X_test = {k: _load_csv(_file_path, k, _test_suffix) for k in modality_keys}
    y_train = _load_csv(_file_path, "outcomes_label", _train_suffix)
    y_test = _load_csv(_file_path, "outcomes_label", _test_suffix)
    dictionary = pd.read_csv(join(_file_path, "data_dictionary.csv"))

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "data_dict": dictionary,
    }


def _load_csv(file_path, fn, suffix):
    return pd.read_csv(join(file_path, f"{fn}_{suffix}.csv"), index_col=0)


def get_data_home(data_home=None):
    """Return the path of the eipy data directory.

    This function is referring from scikit-learn.

    This folder is used by some large dataset loaders to avoid downloading the
    data several times.

    By default the data directory is set to a folder named 'eipy_data' in the
    user home folder.

    Alternatively, it can be set by the 'EIPY_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.

    Parameters
    ----------
    data_home : str or path-like, default=None
        The path to scikit-learn data directory. If `None`, the default path
        is `~/eipy_data`.

    Returns
    -------
    data_home: str
        The path to eipy data directory.
    """
    if data_home is None:
        data_home = environ.get("EIPY_DATA", join("~", "eipy_data"))
    data_home = expanduser(data_home)
    makedirs(data_home, exist_ok=True)
    return data_home
