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

    params_dict = {
        'o.learning_rate': [0.001, 0.0005],
        'o.weight_decay': [0.0, 0.0001],
        'freeze_layers': [True, False],
        't.epochs': [50],
        'ema_decay': [0.],  # 0.999
        'data_args.n_labels': [40, 100, 200, 800, 2000, 4000, 20000],
    }
    params_list = [dict(zip(params_dict.keys(), values)) for values in itertools.product(*params_dict.values())]

    for run_ind, params in enumerate(params_list):
        logger.info(f'================== Run {run_ind+1}/{len(params_list)} ==================')

        # Recalculating number of epochs
        params['early_stopping'] = dict(patience=params['t.epochs'] // 2, monitor='cross_entropy')

        existing_runs = mlflow.search_runs(filter_string=f"params.run_hash = '{calculate_hash(params)}'",
                                           run_view_type=mlflow.tracking.client.ViewType.ACTIVE_ONLY,
                                           experiment_ids=['2'])
        if len(existing_runs) > 0:
            logger.info('Skipping existing run.')
            continue

        params_list = [('--' + k, str(v)) for k, v in params.items()]
        params_list = list(itertools.chain(*params_list))
        call(['python3', f'{SRC_PATH}/models/full_supervised/main.py', '--run_hash', calculate_hash(params)] + sys.argv[1:] + params_list)
