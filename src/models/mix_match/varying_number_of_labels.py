import mlflow
import logging
import hashlib
import itertools
from cortex.main import run
from subprocess import call
import sys

from src.models.mix_match.utils import calculate_hash
from src import SRC_PATH, MLFLOW_SSL_URI

logger = logging.getLogger('ssl_evaluation')
logger.setLevel(logging.INFO)

if __name__ == '__main__':

    mlflow.set_tracking_uri(MLFLOW_SSL_URI)

    params_dict = {
        'o.learning_rate': [0.01],
        't.epochs': [1000],
        'T': [0.75],  # 0.75
        'ema_decay': [0.999],  # 0.999
        'alpha': [0.5],  # 0.5
        'lambda_u': [50],  # 25 - 500 epochs / 50 - 1000 epochs,
        'data_args.n_labels': [40, 100, 200, 800, 2000, 4000, 20000]
    }
    params_list = [dict(zip(params_dict.keys(), values)) for values in itertools.product(*params_dict.values())]

    for run_ind, params in enumerate(params_list):
        logger.info(f'================== Run {run_ind+1}/{len(params_list)} ==================')

        existing_runs = mlflow.search_runs(filter_string=f"params.run_hash = '{calculate_hash(params)}'",
                                           run_view_type=mlflow.tracking.client.ViewType.ACTIVE_ONLY,
                                           experiment_ids=['1'])
        if len(existing_runs) > 0:
            logger.info('Skipping existing run.')
            continue

        params_list = [('--' + k, str(v)) for k, v in params.items()]
        params_list = list(itertools.chain(*params_list))
        call(['python3', f'{SRC_PATH}/models/mix_match/main.py', '--run_hash', calculate_hash(params)] + sys.argv[1:] + params_list)
