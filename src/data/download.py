import kaggle
from src import DATA_PATH


kaggle.api.authenticate()
kaggle.api.dataset_download_files('paultimothymooney/kermany2018', path=DATA_PATH, unzip=True)