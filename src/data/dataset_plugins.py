import torchvision
from cortex.built_ins.datasets.torchvision_datasets import TorchvisionDatasetPlugin
from cortex.plugins import DatasetPlugin, register_plugin
from cortex._lib.data import _PLUGINS, DATA_HANDLER, DATASETS, DataHandler
from cortex._lib import exp
import logging
from src.data.transforms import TransformTwice, TransformFix
import os
from copy import deepcopy
from src import DATA_PATH
from src.data.transforms import build_transforms

import numpy as np
from torchvision.datasets import VisionDataset

logger = logging.getLogger('dataset_plugins')


# Modifying DataHandler methods to generate paired batches
def make_iterator(self, source):
    loader = self.loaders[source][self.mode]

    def iterator():
        for inputs in loader:
            new_inputs = []
            for inp in inputs:
                if isinstance(inp, list):
                    new_inputs.append([sub_inp.to(exp.DEVICE) for sub_inp in inp])
                else:
                    new_inputs.append(inp.to(exp.DEVICE))
            inputs = new_inputs
            inputs_ = []
            for i, inp in enumerate(inputs):
                inputs_.append(inp)
            yield inputs_

    return iterator()


def __next__(self):
    """__next__ function that returns next batch.

    Loops through the sources (as defined by loaders) and draws batches from
    each iterator.

    For each source, we first:
        1) Draw a tuple of data from the iterator
        2) Check if the number of examples is consistent with other sources.
        3) Form a dictionary of input names for the source and the data tuple.
        4) Add this dictionary to a dictionary of outputs.
        5) Add any noise variables.

    Returns:
        tuple: Tuple of batches.

    """
    output = {}
    sources = self.loaders.keys()

    batch_size = self.batch_size[self.mode]
    for source in sources:
        data = next(self.iterators[source])
        current_batch_size = data[0][0].size()[0] if isinstance(data[0], list) else data[0].size()[0]
        if current_batch_size < batch_size:
            if self.skip_last_batch:
                raise StopIteration
            batch_size = current_batch_size
        data = dict((k, v) for k, v in zip(self.input_names[source], data))

        output[source] = data

    for k, n_vars in self.noise.items():
        n_var = n_vars[self.mode]
        n_var = n_var.sample()
        n_var = n_var.to(exp.DEVICE)

        if n_var.size()[0] != batch_size:
            n_var = n_var[0:batch_size]
        output[k] = n_var

    self.batch = output
    self.u += 1
    self.update_pbar()

    if self.mode == 'train':
        exp.INFO['data_steps'] += 1

    return self.batch


def remove_labels(train_set: VisionDataset, n_labels: int):
    labels_attr = None
    if hasattr(train_set, 'labels'):
        labels_attr = 'labels'
    elif hasattr(train_set, 'targets'):
        labels_attr = 'targets'

    labels = np.array(getattr(train_set, labels_attr))
    assert n_labels <= len(labels)  # Requiring more labels, than exist

    classes = np.unique(labels)
    classes = classes[classes != -1]  # For originally SSL datasets
    n_labels_per_cls = n_labels // len(classes)

    if n_labels % len(classes) != 0:
        logger.warning(f'Not equal distribution of classes. Used number of labels: {n_labels_per_cls * len(classes)}')

    for c in classes:
        labels[np.where(labels == c)[0][n_labels_per_cls:]] = -1
    setattr(train_set, labels_attr, labels)


class SSLDatasetPlugin(TorchvisionDatasetPlugin):

    def handle(self,
               # Dataset name and folder
               source: str,
               source_folder: str = None,
               # SSL
               n_labels: int = None,
               split_labelled_and_unlabelled: bool = False,
               fix_match_transform=False,
               labeled_only: bool = False,
               mu: int = None,
               # Preprocessing
               normalize=True,
               train_samples: int = None,
               test_samples: int = None,
               # Train augmentations
               train_transform=None,
               extra_init_train_transform: object = None,
               center_crop: int = None,
               image_size: int = None,
               random_crop: int = None,
               flip: bool = False,
               random_resize_crop: int = None,
               # Test augmentations
               test_transform=None,
               extra_init_test_transform: object = None,
               center_crop_test: int = None,
               image_size_test: int = None,
               random_crop_test: int = None,
               flip_test: bool = False,
               random_resize_crop_test: int = None):
        """
       Args:
           :param fix_match_transform: Use FixMatch transformation instead of TransforTwice
           :param mu: Mu parameter for fix-match
           :param source_folder: Dataset folder in <project_root>/data/, should be specified if source == ImageFolder
           :param extra_init_test_transform:
           :param extra_init_train_transform:
           :param test_transform:
           :param train_transform:
           :param split_labelled_and_unlabelled:
           :param random_resize_crop_test:
           :param flip_test:
           :param random_crop_test:
           :param image_size_test:
           :param center_crop_test:
           :param source:
           :param n_labels: Number of labels
           :param normalize: Normalization of the image.
           :param train_samples: Number of training samples.
           :param test_samples: Number of test samples.
           :param labeled_only: Only use labeled data.
           :param center_crop: Center cropping of the image.
           :param image_size: Final size of the image.
           :param random_crop: Random cropping of the image.
           :param flip: Random flipping.
           :param random_resize_crop: Random resizing and cropping of the image.
       """
        Dataset = getattr(torchvision.datasets, source)
        Dataset = self.make_indexing(Dataset)

        torchvision_path = self.get_path('torchvision')
        if not os.path.isdir(torchvision_path):
            os.mkdir(torchvision_path)

        if normalize and isinstance(normalize, bool):
            if source in ['MNIST', 'dSprites', 'Fashion-MNIST', 'EMNIST', 'PhotoTour']:
                normalize = [(0.5,), (0.5,)]
                scale = (0, 1)
            elif source == 'ImageFolder':
                # specific constants for Pytorch WideResNet
                normalize = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
                scale = (0, 1)
            else:
                normalize = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
                scale = (-1, 1)
        else:
            scale = None

        if train_transform is None or test_transform is None:
            train_transform = build_transforms(normalize=normalize, center_crop=center_crop, image_size=image_size,
                                               random_crop=random_crop, flip=flip, random_resize_crop=random_resize_crop)
            test_transform = build_transforms(normalize=normalize, center_crop=center_crop_test, image_size=image_size_test,
                                              random_crop=random_crop_test, flip=flip_test,
                                              random_resize_crop=random_resize_crop_test)

        if extra_init_train_transform is not None:
            train_transform.transforms.insert(0, extra_init_train_transform)

        if extra_init_test_transform is not None:
            test_transform.transforms.insert(0, extra_init_test_transform)

        if source == 'ImageFolder':
            handler = self._handle_ImageFolder
            data_path = DATA_PATH + '/' + exp.NAME
        else:
            data_path = os.path.join(torchvision_path, source)
            if self.copy_to_local:
                data_path = self.copy_to_local_path(data_path)
            handler = self._handle

        train_set, val_set, test_set = handler(Dataset, data_path, transform=train_transform, test_transform=test_transform,
                                               labeled_only=labeled_only)
        # Removing labels for Semi-Supervised setup
        if n_labels is not None:
            remove_labels(train_set, n_labels)

        dim_images = train_set[0][0].size()

        labels = None
        if hasattr(train_set, 'labels'):
            labels = np.array(train_set.labels)
            labels_attr = 'labels'
        elif hasattr(train_set, 'targets'):
            labels = np.array(train_set.targets)
            labels_attr = 'targets'

        uniques = sorted(np.unique(labels).tolist())

        if -1 in uniques:
            uniques = uniques[1:]

        dim_l = len(uniques)
        dims = dict(images=dim_images, targets=dim_l)
        input_names = ['images', 'targets', 'index']

        self.add_dataset(source, data=dict(train=train_set, val=val_set, test=test_set),
                         input_names=input_names, dims=dims, scale=scale)

        if split_labelled_and_unlabelled:
            DATA_HANDLER.batch_size['train'] //= 2

            train_set_labeled = deepcopy(train_set)
            train_set_labeled.samples = [train_set_labeled.samples[i] for i in range(len(labels)) if labels[i] != -1]
            setattr(train_set_labeled, labels_attr, list(labels[labels != -1]))

            train_set_unlabeled = deepcopy(train_set)
            train_set_unlabeled.samples = [train_set_unlabeled.samples[i] for i in range(len(labels)) if labels[i] == -1]
            setattr(train_set_unlabeled, labels_attr, list(labels[labels == -1]))
            if fix_match_transform:
                train_set_unlabeled.transform = TransformFix(train_set_unlabeled.transform)
            else:
                train_set_unlabeled.transform = TransformTwice(train_set_unlabeled.transform)

            self.add_dataset(source + '_l', data=dict(train=train_set_labeled, val=val_set, test=test_set),
                             input_names=input_names, dims=dims, scale=scale)
            self.add_dataset(source + '_u', data=dict(train=train_set_unlabeled, val=val_set, test=test_set),
                             input_names=input_names, dims=dims, scale=scale)

            DATA_HANDLER.add_dataset(DATASETS, source + '_l', 'data_l', self, n_workers=1, shuffle=True)

            if mu is not None:
                DATA_HANDLER.batch_size['train'] *= mu
            DATA_HANDLER.add_dataset(DATASETS, source + '_u', 'data_u', self, n_workers=1, shuffle=True)
            if mu is not None:
                DATA_HANDLER.batch_size['train'] //= mu  # For early stopping to work

    def _handle(self, Dataset, data_path, transform=None, test_transform=None, **kwargs):
        train_set = Dataset(data_path, train=True, transform=transform, download=True)
        test_set = Dataset(data_path, train=False, transform=test_transform, download=True)
        return train_set, test_set

    def _handle_ImageFolder(self, Dataset, data_path, transform=None, test_transform=None, **kwargs):
        train_set = Dataset(f'{data_path}/train', transform=transform)
        if 'UNL' in train_set.classes:
            train_set.targets = np.array(train_set.targets)
            train_set.targets[train_set.targets == train_set.class_to_idx['UNL']] = -1
            train_set.class_to_idx['UNL'] = -1
        val_set = Dataset(f'{data_path}/val', transform=test_transform)
        test_set = Dataset(f'{data_path}/test', transform=test_transform)
        return train_set, val_set, test_set


# Removing all registered DatasetPlugins
_PLUGINS.clear()
register_plugin(SSLDatasetPlugin)
DataHandler.make_iterator = make_iterator
DataHandler.__next__ = __next__
