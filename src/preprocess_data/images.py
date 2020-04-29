from torchvision import transforms, models
import numpy as np
from enum import Enum
import logging
import torch.nn as nn
import albumentations as A


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

    elif model_type == ImageNetModelsPretrained.VGG16.value:
        logging.info("image model with vgg16 model")
        image_model = models.vgg16(pretrained=True)
        modules = list(image_model.children())[:-1]
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


class ColorsAugmentation(Enum):
    LIGHT = 0
    CLACHE = 1
    RANDOM_CONSTRACT = 2
    RANDOM_GAMMA = 3
    RANDOM_BRIGHTNESS = 4
    GRAY = 5
    JPEG_COMPREENSION = 6
    NO_AUGMENTATION = 7


def apply_no_transformation(image):
    return {"image": image}


def augment_image_with_color():
    mode = np.random.randint(len(ColorsAugmentation))
    mode = 7

    if mode == ColorsAugmentation.LIGHT.value:
        return A.Compose([
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
            A.CLAHE(p=1),
        ], p=1)
    elif mode == ColorsAugmentation.CLACHE.value:
        return A.CLAHE(p=1)
    elif mode == ColorsAugmentation.RANDOM_CONSTRACT.value:
        return A.RandomContrast(p=1)
    elif mode == ColorsAugmentation.RANDOM_GAMMA.value:
        return A.RandomGamma(p=1)
    elif mode == ColorsAugmentation.RANDOM_BRIGHTNESS.value:
        return A.RandomBrightness(p=1)
    elif mode == ColorsAugmentation.GRAY.value:
        return A.ToGray(p=1)
    elif mode == ColorsAugmentation.JPEG_COMPREENSION.value:
        return A.JpegCompression(p=1)
    elif mode == ColorsAugmentation.NO_AUGMENTATION.value:
        return apply_no_transformation
    else:
        raise ValueError(
            "Mode should be equal to 0-6 (see ENUM ColorsAugmentation).")


def augment_image_with_rotations_and_flips():
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


def augment_image():
    mode = np.random.randint(2)
    if mode == 0:
        return augment_image_with_color()
    else:
        return augment_image_with_rotations_and_flips()
