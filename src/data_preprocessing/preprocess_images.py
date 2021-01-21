from torchvision import transforms, models
import numpy as np
from enum import Enum
from utils.enums import ImageNetModelsPretrained
import logging
import torch.nn as nn
import albumentations as A
import torch
from efficientnet_pytorch import EfficientNet
from definitions_datasets import PATH_DATASETS_UCM, PATH_DATASETS_FLICKR8K


class EfficientEmbeddingsNet(nn.Module):
    def __init__(self, embeddings_dim=300):
        super(EfficientEmbeddingsNet, self).__init__()
        self.cnn = EfficientNet.from_pretrained('efficientnet-b5')

        previous_out_channels = self.cnn._conv_head.out_channels
        self._conv11 = nn.Conv2d(previous_out_channels, embeddings_dim, kernel_size=1, bias=False)

    def extract_features(self, inputs):
        features = self.cnn.extract_features(inputs)
        image_regions_embeddings = self._conv11(features)

        return image_regions_embeddings

    def forward(self, inputs):
        image_regions_embeddings = self.extract_features(inputs)
        # Pooling and final linear layer
        x = self.cnn._avg_pooling(image_regions_embeddings)

        x = x.flatten(start_dim=1)

        x = self.cnn._dropout(x)
        x = self.cnn._fc(x)

        return x


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

    elif model_type == ImageNetModelsPretrained.MULTILABEL_ALL_600.value:
        logging.info("image model with densenet model (all) with multi-label classification on modified 600")

        checkpoint = torch.load('experiments/results/classification_densenet_modifiedrsicd_600.pth.tar')
        vocab_size = 600

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

    elif model_type == ImageNetModelsPretrained.EFFICIENCENET_IMAGENET.value:
        # https://github.com/lukemelas/EfficientNet-PyTorch/pull/194
        logging.info("image model with efficientnet model pre-trained on imagenet")

        image_model = EfficientNet.from_pretrained('efficientnet-b5')
        encoder_dim = image_model._fc.in_features

        return image_model, encoder_dim

    elif model_type == ImageNetModelsPretrained.MULTILABEL_ALL_EFFICIENCENET.value:
        # https://github.com/lukemelas/EfficientNet-PyTorch/pull/194
        logging.info("image model with efficientnet model (all) with multi-label classification")

        checkpoint = torch.load('experiments/results/classification_efficientnet_modifiedrsicd.pth.tar')
        vocab_size = 512

        image_model = EfficientNet.from_pretrained('efficientnet-b5')
        encoder_dim = image_model._fc.in_features
        image_model._fc = nn.Linear(encoder_dim, vocab_size)

        image_model.load_state_dict(checkpoint['model'])

        return image_model, encoder_dim

    elif model_type == ImageNetModelsPretrained.EFFICIENCENET_RSICD_NOUNSADJS_EMBEDDINGS.value:
        # https://github.com/lukemelas/EfficientNet-PyTorch/pull/194
        logging.info("image model with efficientnet RSICD multi-label classification for with emb adjs nouns")

        checkpoint = torch.load('experiments/results/classification_efficientnet_rsicd_embedding_nouns_adjs.pth.tar')
        emb_dim = 300

        image_model = EfficientNet.from_pretrained('efficientnet-b5')
        encoder_dim = image_model._fc.in_features
        image_model._fc = nn.Linear(encoder_dim, emb_dim)

        image_model.load_state_dict(checkpoint['model'])

        return image_model, encoder_dim

    elif model_type == ImageNetModelsPretrained.EFFICIENCENET_RSICD_CAPTION_EMBEDDINGS.value:
        # https://github.com/lukemelas/EfficientNet-PyTorch/pull/194
        logging.info("image model with efficientnet RSICD multi-label classification with emb caption")

        checkpoint = torch.load('experiments/results/classification_efficientnet_rsicd_embedding_caption.pth.tar')
        emb_dim = 300

        image_model = EfficientNet.from_pretrained('efficientnet-b5')
        encoder_dim = image_model._fc.in_features
        image_model._fc = nn.Linear(encoder_dim, emb_dim)

        image_model.load_state_dict(checkpoint['model'])

        return image_model, encoder_dim

    elif model_type == ImageNetModelsPretrained.EFFICIENCENET_RSICD_CAPTION_GLOVE_EMBEDDINGS.value:
        # https://github.com/lukemelas/EfficientNet-PyTorch/pull/194
        logging.info("image model with efficientnet RSICD multi-label classification with emb caption glove")

        checkpoint = torch.load('experiments/results/classification_efficientnet_rsicd_embedding_caption_glove.pth.tar')
        emb_dim = 300

        image_model = EfficientNet.from_pretrained('efficientnet-b5')
        encoder_dim = image_model._fc.in_features
        image_model._fc = nn.Linear(encoder_dim, emb_dim)

        image_model.load_state_dict(checkpoint['model'])

        return image_model, encoder_dim

    elif model_type == ImageNetModelsPretrained.EFFICIENCENET_RSICD_CAPTION_GLOVE_EMBEDDINGS_SMOOTHL1.value:
        # https://github.com/lukemelas/EfficientNet-PyTorch/pull/194
        logging.info("image model with efficientnet RSICD multi-label classification with emb caption glove smoorhl1")

        checkpoint = torch.load('experiments/results/classification_efficientnet_rsicd_embedding_caption_glove_smoothl1.pth.tar')
        emb_dim = 300

        image_model = EfficientNet.from_pretrained('efficientnet-b5')
        encoder_dim = image_model._fc.in_features
        image_model._fc = nn.Linear(encoder_dim, emb_dim)

        image_model.load_state_dict(checkpoint['model'])

        return image_model, encoder_dim

    elif model_type == ImageNetModelsPretrained.EFFICIENCENETB7_RSICD_CAPTION_GLOVE_EMBEDDINGS_SMOOTHL1.value:
        # https://github.com/lukemelas/EfficientNet-PyTorch/pull/194
        logging.info("image model with efficientnet B7 RSICD multi-label classification with emb caption glove smoorhl1")

        checkpoint = torch.load('experiments/results/classification_efficientnet_b7_rsicd_embedding_caption_glove_smoothl1.pth.tar')
        emb_dim = 300

        image_model = EfficientNet.from_pretrained('efficientnet-b7')
        encoder_dim = image_model._fc.in_features
        image_model._fc = nn.Linear(encoder_dim, emb_dim)

        print("encoder dim", encoder_dim)

        image_model.load_state_dict(checkpoint['model'])

        return image_model, encoder_dim

    elif model_type == ImageNetModelsPretrained.EFFICIENCENET_RSICD_CAPTION_FASTTEXT_EMBEDDINGS_SMOOTHL1.value:
        # https://github.com/lukemelas/EfficientNet-PyTorch/pull/194
        logging.info("image model with efficientnet RSICD multi-label classification with emb caption fasttext smoorhl1")

        checkpoint = torch.load('experiments/results/classification_efficientnet_rsicd_embedding_caption_fasttext_smoothl1.pth.tar')
        emb_dim = 300

        image_model = EfficientNet.from_pretrained('efficientnet-b5')
        encoder_dim = image_model._fc.in_features
        image_model._fc = nn.Linear(encoder_dim, emb_dim)

        image_model.load_state_dict(checkpoint['model'])

        return image_model, encoder_dim

    elif model_type == ImageNetModelsPretrained.EFFICIENCENET_UCM_CAPTION_GLOVE_EMBEDDINGS_SMOOTHL1.value:
        # https://github.com/lukemelas/EfficientNet-PyTorch/pull/194
        logging.info("image model with efficientnet ucm multi-label classification with emb caption glove smoorhl1")

        checkpoint = torch.load('experiments/results/classification_efficientnet_ucm_embedding_caption_glove_smoothl1.pth.tar')
        emb_dim = 300

        image_model = EfficientNet.from_pretrained('efficientnet-b5')
        encoder_dim = image_model._fc.in_features
        image_model._fc = nn.Linear(encoder_dim, emb_dim)

        image_model.load_state_dict(checkpoint['model'])

        return image_model, encoder_dim

    # elif model_type == ImageNetModelsPretrained.EFFICIENCENET_EMBEDDINGS.value:
    #     # https://github.com/lukemelas/EfficientNet-PyTorch/pull/194
    #     logging.info("image model with efficientnet multi-label classification with conv11")

    #     checkpoint = torch.load('experiments/results/classification_efficientnet_regions.pth.tar')
    #     vocab_size = 512

    #     image_model = EfficientEmbeddingsNet()
    #     encoder_dim = 300
    #     image_model.cnn._fc = nn.Linear(encoder_dim, vocab_size)

    #     image_model.load_state_dict(checkpoint['model'])

    #     return image_model, encoder_dim

    elif model_type == ImageNetModelsPretrained.EFFICIENCENET_UCM.value:
        # https://github.com/lukemelas/EfficientNet-PyTorch/pull/194
        logging.info("image model with UCM efficientnet model (all) with multi-label classification")

        checkpoint = torch.load('experiments/results/classification_efficientnet_ucm.pth.tar')
        classification_data_state = torch.load(PATH_DATASETS_UCM + "classification_dataset_ucm")
        vocab_size = len(classification_data_state["classes_to_id"])
        print("n classes", vocab_size)

        image_model = EfficientNet.from_pretrained('efficientnet-b5')
        encoder_dim = image_model._fc.in_features
        image_model._fc = nn.Linear(encoder_dim, vocab_size)

        image_model.load_state_dict(checkpoint['model'])

        return image_model, encoder_dim

    elif model_type == ImageNetModelsPretrained.EFFICIENCENET_FLICKR8K.value:
        # https://github.com/lukemelas/EfficientNet-PyTorch/pull/194
        logging.info("image model with Flickr8k efficientnet model (all) with multi-label classification")

        checkpoint = torch.load('experiments/results/classification_efficientnet_flickr8k.pth.tar')
        classification_data_state = torch.load(PATH_DATASETS_FLICKR8K + "classification_dataset_flickr8k")
        vocab_size = len(classification_data_state["classes_to_id"])
        print("n classes", vocab_size)

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
    CLACHE = 0
    RANDOM_CONSTRACT = 1
    RANDOM_GAMMA = 2
    RANDOM_BRIGHTNESS = 3
    GRAY = 4
    JPEG_COMPREENSION = 5
    NO_AUGMENTATION = 6


def apply_no_transformation(image):
    return {"image": image}


def augment_image_with_color():
    mode = np.random.randint(len(ColorsAugmentation))

    if mode == ColorsAugmentation.CLACHE.value:
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
