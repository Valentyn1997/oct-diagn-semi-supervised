import kaggle
from src import ROOT_PATH


kaggle.api.authenticate()
kaggle.api.dataset_download_files('paultimothymooney/kermany2018', path=f'{ROOT_PATH}/data', unzip=True)