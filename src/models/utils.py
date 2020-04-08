import re
import mlflow
import torch
from cortex._lib import exp


def remove_wrong_characters(metric_key):
    return re.sub(r'\(|\)|\]|\[|,', '_', metric_key)


class MlflowLogger:
    log_to_mlflow = False

    @staticmethod
    def start_run(experiment_name, checkpoints_to_mlflow=False):
        MlflowLogger.log_to_mlflow = True
        MlflowLogger.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        MlflowLogger.run = mlflow.start_run()
        if checkpoints_to_mlflow:
            exp.setup_out_dir(mlflow.get_artifact_uri(), None, None)

    @staticmethod
    def end_run():
        mlflow.end_run()

    @staticmethod
    def log_basic_run_params(input_shape, output_shape=None):
        mlflow.log_param('batch_size', exp.ARGS['data']['batch_size'])
        mlflow.log_param('input_shape', input_shape)
        if output_shape is not None:
            mlflow.log_param('output_shape', output_shape)
        mlflow.log_param('optimizer', exp.ARGS['optimizer']['optimizer'])
        mlflow.log_param('lr', exp.ARGS['optimizer']['learning_rate'])
        mlflow.log_param('weight_decay', exp.ARGS['optimizer']['weight_decay'])
        mlflow.log_param('epochs', exp.ARGS['train']['epochs'])
        mlflow.log_param('normalize', exp.ARGS['data']['data']['normalize'])
        mlflow.log_param('shuffle', exp.ARGS['data']['shuffle'])
        mlflow.log_param('skip_last_batch', exp.ARGS['data']['skip_last_batch'])

    @staticmethod
    def log_ssl_parameters():
        mlflow.log_param('split_lab_unlab', exp.ARGS['data']['data']['split_labelled_and_unlabelled'])
        mlflow.log_param('n_labels_train', exp.ARGS['data']['data']['n_labels'])
        mlflow.log_param('augmentation', exp.ARGS['data']['data'])

    @staticmethod
    def log_all_metrics(mode='test', epoch_to_log=None):
        if MlflowLogger.log_to_mlflow:
            epoch_to_log = exp.INFO['epoch'] if epoch_to_log is None else epoch_to_log
            losses = exp.RESULTS.pull_all(mode, 'losses', epoch=epoch_to_log)
            other_metrics = exp.RESULTS.pull_all(mode, 'results', epoch=epoch_to_log)
            metrics = {}
            for k, v in {**losses, **other_metrics}.items():
                if mode == 'test':
                    suffix = ''
                else:
                    suffix = f'_{mode}'
                metrics[remove_wrong_characters(k) + suffix] = v
            mlflow.log_metrics(metrics, step=exp.INFO['epoch'])


class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * exp.ARGS['optimizer']['learning_rate']

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if len(param.shape) > 0:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)


class EarlyStopping(object):
    def __init__(self, monitor='acc_top1', mode='val', min_delta=0.5, patience=5):
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.best_value = 0.0
        self.stopped_epoch = None

    def on_epoch_end(self):
        current_value = exp.RESULTS.pull_all(self.mode, 'results', epoch=exp.INFO['epoch'])[self.monitor]
        if current_value is None:
            pass
        else:
            if (current_value - self.best_value) > self.min_delta:
                self.best_value = current_value
                self.wait = 1
            else:
                if self.wait >= self.patience:
                    self.stopped_epoch = exp.INFO['epoch']
                self.wait += 1



def accuracy(outputs, targets, labeled, top: int = 1, is_argmax=False):
    """Computes the accuracy.

    Args:
        :param is_argmax:
        :param targets: Targets for each input.
        :param labeled: Binary variable indicating whether a target exists.
        :param top: Top-K accuracy.
        :param outputs: Classifier outputs.

    """
    with torch.no_grad():
        if is_argmax:
            pred = outputs.unsqueeze(1)
        else:
            _, pred = outputs.topk(top, 1, True, True)
        pred = pred.t()
        correct = labeled.float() * pred.eq(targets.view(1, -1).expand_as(pred)).float()

        correct_k = correct[:top].view(-1).float().sum(0, keepdim=True)
        accuracy = correct_k.mul_(100.0 / labeled.float().sum())
        return accuracy.detach().item()