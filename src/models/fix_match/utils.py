from torch.optim.lr_scheduler import LambdaLR
import math


class CosineLRPolicy(object):
    def __init__(self, num_warmup_steps, num_training_steps, num_cycles=7. / 16.):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles

    def __call__(self, epoch):
        if epoch < self.num_warmup_steps:
            return float(epoch) / float(max(1, self.num_warmup_steps))
        no_progress = float(epoch - self.num_warmup_steps) / \
                      float(max(1, self.num_training_steps - self.num_warmup_steps))
        return max(0., math.cos(math.pi * self.num_cycles * no_progress))

    def __repr__(self):
        return str(vars(self))

    def __str__(self):
        return str(vars(self))

