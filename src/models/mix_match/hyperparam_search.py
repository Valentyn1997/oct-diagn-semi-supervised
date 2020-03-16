import mlflow
import logging
import hashlib
import itertools
from cortex.main import run

from src.models.mix_match.controller import MixMatchController
from src.data.dataset_plugins import SSLDatasetPlugin
from src import MLFLOW_SSL_URI
from cortex._lib import exp
from src.models.mix_match.utils import calculate_hash

from src.models.mix_match.utils import _to_dot

logger = logging.getLogger('ssl_evaluation')
logger.setLevel(logging.INFO)

if __name__ == '__main__':

    mlflow.set_tracking_uri(MLFLOW_SSL_URI)

    search_dict = {
        'o.learning_rate': [0.05, 0.01, 0.002, 0.001],
        't.epochs': [500, 1000, 2000],
        'T': [0.25, 0.5, 0.75],
        'ema_decay': [0.9, 0.99, 0.999],
        'alpha': [0.5, 0.75, 0.9],
        'lambda_u': [25, 50, 100, 150]
    }
    search_list = [dict(zip(search_dict.keys(), values)) for values in itertools.product(*search_dict.values())]


    for run_ind, params in enumerate(search_list):
        logger.info(f'================== Run {run_ind+1}/{len(search_list)} ==================')

        existing_runs = mlflow.search_runs(filter_string=f"params.run_hash = '{calculate_hash(params)}'",
                                           run_view_type=mlflow.tracking.client.ViewType.ACTIVE_ONLY,
                                           experiment_ids=[1, 2])
        if len(existing_runs) > 0:
            logger.info('Skipping existing run.')
            continue

        controller = MixMatchController(params)

        # Running experiment
        run(model=controller)
