import mlflow
import logging
import itertools
from subprocess import call
import sys

from src.models.utils import calculate_hash
from src import SRC_PATH, MLFLOW_SSL_URI

logger = logging.getLogger('ssl_evaluation')
logger.setLevel(logging.INFO)

if __name__ == '__main__':

    mlflow.set_tracking_uri(MLFLOW_SSL_URI)

    search_dict = {
        'o.learning_rate': [0.01, 0.001],  # 0.01
        't.epochs': [500, 1000],  # 1000
        'T': [0.25, 0.5, 0.75, 0.9],  # 0.5
        'ema_decay': [0.9, 0.999],  # 0.999
        'alpha': [0.25, 0.5, 0.75, 0.9],  # 0.9
        'lambda_u': [12.5, 25, 50, 100, 150]  # 25
    }
    search_list = [dict(zip(search_dict.keys(), values)) for values in itertools.product(*search_dict.values())]

    for run_ind, params in enumerate(search_list):
        logger.info(f'================== Run {run_ind+1}/{len(search_list)} ==================')

        existing_runs = mlflow.search_runs(filter_string=f"params.run_hash = '{calculate_hash(params)}'",
                                           run_view_type=mlflow.tracking.client.ViewType.ACTIVE_ONLY,
                                           experiment_ids=['1', '2'])
        if len(existing_runs) > 0:
            logger.info('Skipping existing run.')
            continue

        params_list = [('--' + k, str(v)) for k, v in params.items()]
        params_list = list(itertools.chain(*params_list))
        call(['python3', f'{SRC_PATH}/models/mix_match/main.py', '--run_hash', calculate_hash(params)] +
             sys.argv[1:] + params_list)
