from os.path import abspath, dirname
import logging

logging.basicConfig(level=logging.INFO)

ROOT_PATH = dirname(dirname(abspath(__file__)))
DATA_PATH = f'{ROOT_PATH}/data/OCT2017 '
