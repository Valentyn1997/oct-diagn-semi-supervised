import numpy as np
from PIL import Image
from src.data.rand_augment import RandAugmentMC
import torchvision.transforms as transforms


def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')


class RandomPadandCrop(object):
    """Crop randomly the image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, width=4, output_size=None):
        self.width = width
        if output_size is None:
            self.output_size = output_size
        # assert isinstance(output_size, (int, tuple))
        elif isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        old_h, old_w = x.size[:2]
        x = np.transpose(x, (2, 0, 1))
        x = pad(x, self.width)

        h, w = x.shape[1:]
        if self.output_size is None:
            new_h, new_w = old_h, old_w
        else:
            new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[:, top: top + new_h, left: left + new_w]

        return Image.fromarray(np.transpose(x, (1, 2, 0)))


# TODO Implement TransformKTimes
class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class TransformFix(object):
    def __init__(self, base_transform):
        self.weak = base_transform

        # Inserting strong augmentation
        self.strong = []
        for transform in base_transform.transforms:
            if isinstance(transform, transforms.ToTensor):
                self.strong.append(RandAugmentMC(n=2, m=10))
            self.strong.append(transform)
        self.strong = transforms.Compose(self.strong)

    def __call__(self, inp):
        weak = self.weak(inp)
        strong = self.strong(inp)
        return weak, strong


def build_transforms(normalize=None, center_crop=None, image_size=None,
                     random_crop=None, flip=None, random_resize_crop=None):
    """

    Args:
        normalize (tuple or transforms.Normalize): Parameters for data normalization.
        center_crop (int): Size for center crop.
        image_size (int): Size for image size.
        random_crop (int): Size for image random crop.
        flip (bool): Randomly flip the data horizontally.
        random_resize_crop (dict): Random resize crop the image.

    Returns:
        Transforms

    """

    transform_ = []

    if image_size:
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        transform_.append(transforms.Resize(image_size))

    if random_resize_crop:
        transform_.append(transforms.RandomResizedCrop(random_resize_crop['size'], random_resize_crop['scale']))
    elif random_crop:
        transform_.append(transforms.RandomCrop(random_crop))
    elif center_crop:
        transform_.append(transforms.CenterCrop(center_crop))

    if flip:
        transform_.append(transforms.RandomHorizontalFlip())

    transform_.append(transforms.ToTensor())

    if normalize:
        if isinstance(normalize, transforms.Normalize):
            transform_.append(normalize)
        else:
            transform_.append(transforms.Normalize(*normalize))
    transform = transforms.Compose(transform_)
    return transform
