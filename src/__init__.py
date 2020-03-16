from os.path import abspath, dirname
import logging

logging.basicConfig(level=logging.INFO)

ROOT_PATH = dirname(dirname(abspath(__file__)))
DATA_PATH = f'{ROOT_PATH}/data/OCT2017 /OCT2017 '
MLFLOW_SSL_URI = 'http://127.0.0.1:5000'