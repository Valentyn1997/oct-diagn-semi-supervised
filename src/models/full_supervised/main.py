import mlflow
import logging
from cortex.main import run

from src.models.full_supervised.controller import FullSupervisedController
from src.data.dataset_plugins import SSLDatasetPlugin
from src import MLFLOW_SSL_URI

logger = logging.getLogger('ssl_evaluation')

if __name__ == '__main__':

    # if exp.ARGS
    mlflow.set_tracking_uri(MLFLOW_SSL_URI)
    controller = FullSupervisedController()

    run(model=controller)
