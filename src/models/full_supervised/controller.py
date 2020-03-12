import torch
import mlflow
from cortex.plugins import ModelPlugin
from cortex._lib import exp
from torch.backends import cudnn
import torch.nn.functional as F
from torchsummary import summary

from src.models.utils import accuracy, MlflowLogger, WeightEMA
from src.models.wideresnet import WideResNet_50_2


class FullSupervisedController(ModelPlugin):
    defaults = dict(
        data=dict(batch_size=dict(train=32, test=32), inputs=dict(inputs='images'), shuffle=True, skip_last_batch=True),
        train=dict(save_on_lowest='losses.classifier', epochs=25, archive_every=25),
        optimizer=dict(optimizer='Adam', learning_rate=0.01, single_optimizer=True)
    )

    def optimizer_step(self, retain_graph=False):
        super().optimizer_step(retain_graph)
        self.ema_optimizer.step()

    def routine(self, T: float = 0.5, alpha: float = 0.75, *args, **kwargs):
        """
        :param alpha: Parameter of beta distribution
        :param T: Sharpening temperature
        """

        targets = self.inputs('data.targets')
        inputs = self.inputs('data.images')

        outputs = self.nets.classifier(inputs) if self.data.mode == 'train' else self.nets.ema_classifier(inputs)

        cross_entropy = self.criterion(outputs, targets)
        self.losses.classifier = cross_entropy
        self.add_results(cross_entropy=cross_entropy)

        # Top-k accuracy
        with torch.no_grad():
            labeled = 1 - targets.eq(-1).long()
            top1 = accuracy(outputs, targets, labeled, top=1)
            self.add_results(acc_top1=top1)

    def build(self, pretrained=False, ema_decay: float = 0.999, log_to_mlflow=True, *args, **kwargs):
        """
        :param pretrained: Use pretrained on ImageNet encoder, freezing all the layers except last
        :param ema_decay: Exponential moving average decay rate
        :param log_to_mlflow: Log run to mlflow
        """
        cudnn.benchmark = True

        # Reset the data iterator and draw batch to perform shape inference.
        self.data.reset(mode='test', make_pbar=False)
        self.data.next()
        input_shape = self.get_dims('data.images')

        self.nets.classifier = WideResNet_50_2(num_classes=self.get_dims('data.targets'), pretrained=pretrained)
        self.nets.ema_classifier = WideResNet_50_2(num_classes=self.get_dims('data.targets'), pretrained=pretrained)
        print(summary(self.nets.classifier, input_shape))

        self.ema_optimizer = WeightEMA(self.nets.classifier, self.nets.ema_classifier, alpha=ema_decay)
        self.criterion = torch.nn.CrossEntropyLoss()

        if log_to_mlflow:
            MlflowLogger.start_run(exp.INFO['name'] + '_FullSupervised')
            MlflowLogger.log_basic_run_params(input_shape)
            MlflowLogger.log_ssl_parameters()
            mlflow.log_param('pretrained', pretrained)
            mlflow.log_param('ema_decay', ema_decay)

    def eval_loop(self):
        super().eval_loop()
        if MlflowLogger.log_to_mlflow:
            MlflowLogger.log_all_metrics(mode='train')
            MlflowLogger.log_all_metrics(mode='test')

    def visualize(self):
        inputs = self.inputs('data.images')
        targets = self.inputs('data.targets')

        inputs = inputs[:, :1, :, :]
        inputs *= 0.229
        inputs += 0.485

        self.add_image(F.adaptive_avg_pool2d(inputs, (64, 64)), name='Input', labels=targets)
