import numpy as np
import torch
import torchvision.transforms as tf
import torchvision.transforms.functional as F


class BatchRandomCrop(tf.RandomCrop):
    """RandomCrop for a batch of frames with the SAME params."""
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    @staticmethod
    def get_params(img_size,
                   output_size,
                   scale_factor):
        (h, w) = img_size
        # w, h = int(w * scale_factor), int(h * scale_factor)
        th, tw = output_size
        tw_resize, th_resize = int(tw / scale_factor) + 1, int(
            th / scale_factor) + 1

        if h + 1 < th_resize or w + 1 < tw_resize:
            raise ValueError(
                'Required crop size {} is larger then input image size {}'.
                format((th, tw), (h, w)))

        if w == tw_resize and h == th_resize:
            return 0, 0, h, w

        i = torch.randint(0, h - th_resize + 1, size=(1, )).item()
        j = torch.randint(0, w - tw_resize + 1, size=(1, )).item()

        return int(i * scale_factor), int(j * scale_factor), th, tw

    def forward(self, img, params):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = F._get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        return F.crop(img, *params)


class NoneTransform(object):
    def __init__(self):
        pass

    def get_params(self, *args, **kargs):
        return None

    def __call__(self, x, *args, **kargs):
        return x


def do_fal_color_aug(image, gamma_f, bright_f, cbright_f):
    """Do color augmentations as done in FAL-Net."""
    image = image**gamma_f
    image = image * bright_f
    image[image > 1] = 1
    for c in range(3):
        image[c, ...] = image[c, ...] * cbright_f[c]
    image[image > 1] = 1

    return image
