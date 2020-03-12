from torchvision import transforms, models
import numpy as np
from enum import Enum
import logging
import torch.nn as nn


class ImageNetModelsPretrained(Enum):
    RESNET = "resnet"
    DENSENET = "densenet"
    VGG16 = "vgg16"


def get_image_model(model_type):
    if model_type == ImageNetModelsPretrained.RESNET.value:
        logging.info("image model with resnet model")
        image_model = models.resnet101(pretrained=True)
        modules = list(image_model.children())[:-2]
        encoder_dim = 2048
        print("res module -1", list(image_model.children())[:-2])

    elif model_type == ImageNetModelsPretrained.VGG16.value:
        logging.info("image model with vgg16 model")
        image_model = models.vgg16(pretrained=True)
        modules = list(image_model.children())[:-1]
        print("module -1", list(image_model.children())[-2])
        encoder_dim = 512

    else:
        logging.info("image model with densenet model")
        image_model = models.densenet201(pretrained=True)
        modules = list(image_model.children())[:-1]
        encoder_dim = 1920

    return nn.Sequential(*modules), encoder_dim


class FlipsAndRotations(Enum):
    FLIP_HORIZONTAL = 0
    FLIP_VERTICAL = 1
    FLIP_DIAGONAL = 2
    ROT_90 = 3
    ROT_180 = 4
    ROT_270 = 5
    ROT_360 = 6


class MyRotationTransform:
    """Rotate by a given angle."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return transforms.functional.rotate(x, self.angle)


def augment_image():
    mode = np.random.randint(len(FlipsAndRotations))

    if mode == FlipsAndRotations.FLIP_HORIZONTAL.value:
        return transforms.RandomHorizontalFlip(p=1)
    elif mode == FlipsAndRotations.FLIP_VERTICAL.value:
        return transforms.RandomVerticalFlip(p=1)
    elif mode == FlipsAndRotations.FLIP_DIAGONAL.value:
        return transforms.Compose([transforms.RandomHorizontalFlip(p=1),
                                   transforms.RandomVerticalFlip(p=1)])
    elif mode == FlipsAndRotations.ROT_90.value:
        return MyRotationTransform(angle=90)
    elif mode == FlipsAndRotations.ROT_180.value:
        return MyRotationTransform(angle=180)
    elif mode == FlipsAndRotations.ROT_270.value:
        return MyRotationTransform(angle=270)
    elif mode == FlipsAndRotations.ROT_360.value:
        return MyRotationTransform(angle=360)
    else:
        raise ValueError(
            "Mode should be equal to 0-6 (see ENUM FlipsAndRotations).")
