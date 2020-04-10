import torch
import mlflow
from cortex.plugins import ModelPlugin
from cortex._lib import exp
from src.models.mix_match.controller import MixMatchController
from torch.backends import cudnn
import torch.nn.functional as F
from torchsummary import summary

from src.models.utils import accuracy, MlflowLogger, WeightEMA, EarlyStopping
from src.models.wideresnet import WideResNet_50_2


class FullSupervisedController(MixMatchController):
    defaults = dict(
        data=dict(batch_size=dict(train=64, val=32, test=32), inputs=dict(inputs='images'), shuffle=True, skip_last_batch=True),
        train=dict(save_on_lowest='losses.classifier', epochs=25, archive_every=25),
        optimizer=dict(optimizer='Adam', learning_rate=0.01, single_optimizer=True)
    )

    def routine(self, *args, **kwargs):
        source = 'data_l' if exp.ARGS['data']['data']['split_labelled_and_unlabelled'] else 'data'

        targets = self.inputs(f'{source}.targets')
        inputs = self.inputs(f'{source}.images')

        outputs = self.nets.classifier(inputs) if self.data.mode == 'train' else self.nets.ema_classifier(inputs)

        cross_entropy = self.criterion(outputs, targets)
        self.losses.classifier = cross_entropy
        self.add_results(cross_entropy=cross_entropy)

        # Top-k accuracy
        with torch.no_grad():
            labeled = 1 - targets.eq(-1).long()
            top1 = accuracy(outputs, targets, labeled, top=1)
            self.add_results(acc_top1=top1)

    def build(self, pretrained=False, freeze_layers=None, ema_decay: float = 0.999, log_to_mlflow=True,
              early_stopping_patience: int = 20, type_of_run=None, run_hash=None, *args, **kwargs):
        """
        :param run_hash: MD5 hash of hyperparameters string for effective hyperparameter search
        :param type_of_run: Type of run to log to mlflow (as a tag): hyperparam_search, varying_number_of_labels, None
        :param early_stopping_patience: Patience for early stopping, number of epochs
        :param freeze_layers: Freeze all the layers except FC
        :param pretrained: Use pretrained on ImageNet encoder, freezing all the layers except last
        :param ema_decay: Exponential moving average decay rate
        :param log_to_mlflow: Log run to mlflow
        """
        cudnn.benchmark = True

        # Reset the data iterator and draw batch to perform shape inference.
        self.data.reset(mode='test', make_pbar=False)
        self.data.next()
        input_shape = self.get_dims('data.images')

        self.nets.classifier = WideResNet_50_2(num_classes=self.get_dims('data.targets'), pretrained=pretrained,
                                               freeze_layers=freeze_layers)
        self.nets.ema_classifier = WideResNet_50_2(num_classes=self.get_dims('data.targets'), pretrained=pretrained,
                                                   freeze_layers=freeze_layers)
        print(summary(self.nets.classifier, input_shape))

        self.ema_optimizer = WeightEMA(self.nets.classifier, self.nets.ema_classifier, alpha=ema_decay)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.early_stopping = EarlyStopping(patience=exp.ARGS['model']['early_stopping_patience'], min_delta=0.0)

        if log_to_mlflow:
            MlflowLogger.start_run(exp.INFO['name'] + '_FullSupervised')
            MlflowLogger.log_basic_run_params(input_shape)
            MlflowLogger.log_ssl_parameters()
            mlflow.set_tag('type_of_run', type_of_run)
            mlflow.log_param('pretrained', pretrained)
            mlflow.log_param('freeze_layers', freeze_layers)
            mlflow.log_param('ema_decay', ema_decay)
            mlflow.log_param('run_hash', exp.ARGS['model']['run_hash'])
            mlflow.log_param('early_stopping_patience', exp.ARGS['model']['early_stopping_patience'])
