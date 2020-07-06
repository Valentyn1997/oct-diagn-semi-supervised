import mlflow
import torch
from cortex._lib import exp
from torch.backends import cudnn
import torch.nn.functional as F
from torch.nn import DataParallel
import numpy as np
from torchsummary import summary

from src.models.mix_match.controller import MixMatchController
from src.models.utils import accuracy, MlflowLogger, WeightEMA, EarlyStopping, f1_score
from src.models.fix_match.utils import CosineLRPolicy
from src.models.wideresnet import WideResNet_50_2


class FixMatchController(MixMatchController):
    defaults = dict(
        data=dict(batch_size=dict(train=32, val=32, test=64), inputs=dict(inputs='images'), shuffle=True, skip_last_batch=False),
        train=dict(save_on_lowest=None, epochs=500, archive_every=None),
        optimizer=dict(optimizer='SGD', learning_rate=0.03, single_optimizer=True,
                       optimizer_options=dict(momentum=0.9, nesterov=True),
                       scheduler='LambdaLR',
                       scheduler_options=dict(lr_lambda=CosineLRPolicy(0, 500), last_epoch=-1))
    )

    def routine(self, threshold=0.7, lambda_u=5.0, *args, **kwargs):
        """
        :param threshold: Pseudo label threshold
        :param lambda_u: Unlabeled loss weight
        """
        if self.data.mode == 'test' or self.data.mode == 'val':
            targets_l = self.inputs('data.targets')
            inputs_l = self.inputs('data.images')

            self.losses.classifier = torch.tensor(0.0).to(exp.DEVICE)

            with torch.no_grad():
                outputs_l = self.nets.ema_classifier(inputs_l)

        else:
            targets_l = self.inputs('data_l.targets')
            inputs_l = self.inputs('data_l.images')
            inputs_uw, inputs_us = self.inputs('data_u.images')

            if exp.ARGS['data']['data']['mu'] * inputs_l.shape[0] != inputs_uw.shape[0]:
                raise StopIteration

            # Supervised loss
            logits_l = self.nets.classifier(inputs_l)
            Ll = F.cross_entropy(logits_l, targets_l, reduction='mean')

            # Unsupervised loss
            logits_us = self.nets.classifier(inputs_us)
            with torch.no_grad():
                logits_uw = self.nets.classifier(inputs_uw)
                pseudo_label = torch.softmax(logits_uw.detach_(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(threshold).float()
            Lu = (F.cross_entropy(logits_us, targets_u, reduction='none') * mask).mean()

            # record loss
            self.losses.classifier = Ll + lambda_u * Lu

            # Write res
            self.add_results(losses_l=Ll.item())
            self.add_results(losses_u=Lu.item())

            outputs_l = logits_l.detach()

        # Cross-entropy
        with torch.no_grad():
            cross_entropy = self.criterion(outputs_l, targets_l)
            self.add_results(cross_entropy=cross_entropy)

            # Top-k accuracy
            labeled = 1 - targets_l.eq(-1).long()
            top1 = accuracy(outputs_l, targets_l, labeled, top=1)
            # top5 = accuracy(outputs_l, targets_l, labeled, top=5)
            self.add_results(acc_top1=top1)

            # F1-score
            f1 = f1_score(outputs_l, targets_l)
            self.add_results(f1_score=f1)

    def build(self, ema_decay: float = 0.999, early_stopping: dict = None, run_hash=None, log_to_mlflow=True,
              type_of_run=None, *args, **kwargs):
        """
        :param early_stopping: Parameters for early stopping
        :param type_of_run: Type of run to log to mlflow (as a tag): hyperparam_search, varying_number_of_labels, None
        :param run_hash: MD5 hash of hyperparameters string for effective hyperparameter search
        :param log_to_mlflow: Log run to mlflow
        :param ema_decay: Exponential moving average decay rate
        """
        cudnn.benchmark = True

        # Reset the data iterator and draw batch to perform shape inference.
        self.data.reset(mode='test', make_pbar=False)
        self.data.next()
        input_shape = self.get_dims('data.images')

        classifier = WideResNet_50_2(num_classes=self.get_dims('data.targets'), pretrained=False)
        ema_classifier = WideResNet_50_2(num_classes=self.get_dims('data.targets'), pretrained=False)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            exp.DEVICE_IDS = [i for i in range(torch.cuda.device_count())]
            self.nets.classifier = DataParallel(classifier)
            self.nets.ema_classifier = DataParallel(ema_classifier)
        else:
            self.nets.classifier = classifier
            self.nets.ema_classifier = ema_classifier
        print(summary(self.nets.classifier, input_shape))

        for param in self.nets.ema_classifier.parameters():
            param.detach_()

        self.ema_optimizer = WeightEMA(self.nets.classifier, self.nets.ema_classifier, alpha=exp.ARGS['model']['ema_decay'])
        if early_stopping is not None:
            self.early_stopping = EarlyStopping(**early_stopping)
        self.criterion = torch.nn.CrossEntropyLoss()

        if log_to_mlflow:
            MlflowLogger.start_run(exp.INFO['name'] + '_FixMatch')
            MlflowLogger.log_basic_run_params(input_shape)
            MlflowLogger.log_ssl_parameters()
            mlflow.set_tag('type_of_run', type_of_run)
            mlflow.log_param('ema_decay', exp.ARGS['model']['ema_decay'])
            mlflow.log_param('lambda_u', exp.ARGS['model']['lambda_u'])
            mlflow.log_param('threshold', exp.ARGS['model']['threshold'])
            mlflow.log_param('mu', exp.ARGS['data']['data']['mu'])
            mlflow.log_param('run_hash', exp.ARGS['model']['run_hash'])
            mlflow.log_param('early_stopping', exp.ARGS['model']['early_stopping'])

    def visualize(self):
        if exp.ARGS['data']['data']['split_labelled_and_unlabelled'] and self.data.mode == 'train':
            inputs_l = self.inputs('data_l.images')
            targets_l = self.inputs('data_l.targets')

            inputs_uw, inputs_us = self.inputs('data_u.images')
            targets_u = torch.full(targets_l.shape, -1, dtype=torch.long).to(targets_l.device)

            inputs = torch.cat([inputs_l[:12], inputs_uw[:12], inputs_us[:12]])
            targets = torch.cat([targets_l[:12], targets_u[:12], targets_u[:12]])
        else:
            inputs = self.inputs('data.images')
            targets = self.inputs('data.targets')

        inputs = inputs[:, :1, :, :]
        inputs *= 0.229
        inputs += 0.485

        self.add_image(F.adaptive_avg_pool2d(inputs, (64, 64)), name='Input', labels=targets)
