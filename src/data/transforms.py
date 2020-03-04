import numpy as np
from PIL import Image


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