from torchvision import transforms, models
import numpy as np
from enum import Enum
from utils.enums import ImageNetModelsPretrained
import logging
import torch.nn as nn
import albumentations as A
import torch
from efficientnet_pytorch import EfficientNet


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

    elif model_type == ImageNetModelsPretrained.DENSENET.value:
        logging.info("image model with densenet model")
        image_model = models.densenet201(pretrained=True)
        modules = list(image_model.children())[:-1]
        encoder_dim = 1920

    elif model_type == ImageNetModelsPretrained.MULTILABEL_ALL.value:
        logging.info("image model with densenet model (all) with multi-label classification on modified")

        checkpoint = torch.load('experiments/results/classification_densenet_modifiedrsicd.pth.tar')
        vocab_size = 512

        image_model = models.densenet201(pretrained=True)
        encoder_dim = image_model.classifier.in_features
        image_model.classifier = nn.Linear(encoder_dim, vocab_size)

        image_model.load_state_dict(checkpoint['model'])

        modules = list(image_model.children())[:-1]

    elif model_type == ImageNetModelsPretrained.MULTILABEL_LAST.value:
        logging.info("image model with densenet model (last) with multi-label classification")

        checkpoint = torch.load('experiments/results/classification_finetune.pth.tar')
        vocab_size = 512

        image_model = models.densenet201(pretrained=True)
        image_model.classifier = nn.Linear(image_model.classifier.in_features, vocab_size)

        image_model.load_state_dict(checkpoint['model'])

        encoder_dim = 512
        return image_model, encoder_dim

    elif model_type == ImageNetModelsPretrained.MULTILABEL_ALL_EFFICIENCENET.value:
        logging.info("image model with efficientnet model (all) with multi-label classification")

        checkpoint = torch.load('experiments/results/classification_efficientnet_modifiedrsicd.pth.tar')
        vocab_size = 512

        image_model = EfficientNet.from_pretrained('efficientnet-b5')
        encoder_dim = image_model._fc.in_features
        image_model._fc = nn.Linear(encoder_dim, vocab_size)

        image_model.load_state_dict(checkpoint['model'])

        return image_model, encoder_dim

    return nn.Sequential(*modules), encoder_dim


def get_image_extractor(model_type, enable_fine_tuning):
    if model_type == ImageNetModelsPretrained.DENSENET.value or model_type == ImageNetModelsPretrained.MULTILABEL_ALL.value:
        logging.info("image model with densenet model")
        vocab_size = 512

        image_model = models.densenet201(pretrained=True)
        image_model.classifier = nn.Linear(image_model.classifier.in_features, vocab_size)

        encoder_dim = image_model.classifier.in_features

        if model_type == ImageNetModelsPretrained.MULTILABEL_ALL.value:
            logging.info("image model with densenet model (all) with multi-label classification")
            checkpoint = torch.load('experiments/results/classification_densenet_modifiedrsicd.pth.tar')
            image_model.load_state_dict(checkpoint['model'])

        return DenseNetFeatureAndAttrExtractor(image_model), encoder_dim
    else:
        raise Exception("not implemented extractor for the other types")


class DenseNetFeatureAndAttrExtractor(nn.Module):
    def __init__(self, image_model):
        super().__init__()
        self.image_model = image_model

    def forward(self, x):
        modules_features = list(self.image_model.children())[:-1]
        features_extractor = nn.Sequential(*modules_features)

        features = features_extractor(x)
        attrs = self.image_model(x)

        return features, attrs


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


class FlipsAndRotations(Enum):
    FLIP_HORIZONTAL = 0
    FLIP_VERTICAL = 1
    TRANSPOSE = 2
    ROT_90 = 3
    ROT_180 = 4
    ROT_270 = 5
    NO_AUGMENTATION = 6


def augment_image_with_rotations_and_flips():
    mode = np.random.randint(len(FlipsAndRotations))

    if mode == FlipsAndRotations.FLIP_HORIZONTAL.value:
        return A.HorizontalFlip(p=1)
    elif mode == FlipsAndRotations.FLIP_VERTICAL.value:
        return A.VerticalFlip(p=1)
    elif mode == FlipsAndRotations.TRANSPOSE.value:
        return A.Transpose(p=1)
    elif mode == FlipsAndRotations.ROT_90.value:
        return A.Rotate(limit=90, p=1)
    elif mode == FlipsAndRotations.ROT_180.value:
        return A.Rotate(limit=180, p=1)
    elif mode == FlipsAndRotations.ROT_270.value:
        return A.Rotate(limit=270, p=1)
    elif mode == FlipsAndRotations.NO_AUGMENTATION.value:
        return apply_no_transformation
    else:
        raise ValueError(
            "Mode should be equal to 0-6 (see ENUM FlipsAndRotations).")


def augment_image():
    mode = np.random.randint(2)
    if mode == 0:
        return augment_image_with_color()
    else:
        return augment_image_with_rotations_and_flips()
