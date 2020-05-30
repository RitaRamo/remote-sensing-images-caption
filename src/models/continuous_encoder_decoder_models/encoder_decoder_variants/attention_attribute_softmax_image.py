import torchvision
from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from models.abtract_model import AbstractEncoderDecoderModel
import torch.nn.functional as F
from embeddings.embeddings import get_embedding_layer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from preprocess_data.tokens import OOV_TOKEN
from embeddings.embeddings import EmbeddingsType
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_image import ContinuousAttentionImageModel, ContinuousDecoderWithAttentionAndImage
from embeddings.embeddings import EmbeddingsType
from preprocess_data.images import ImageNetModelsPretrained
import logging
from torchvision import models

# chamar image models


def get_image_extractor(model_type, enable_fine_tuning):
    if model_type == ImageNetModelsPretrained.DENSENET.value or model_type == ImageNetModelsPretrained.MULTILABEL_ALL.value:
        logging.info("image model with densenet model")
        vocab_size = 512

        image_model = models.densenet201(pretrained=True)
        image_model.classifier = nn.Linear(image_model.classifier.in_features, vocab_size)

        encoder_dim = image_model.classifier.in_features

        if model_type == ImageNetModelsPretrained.MULTILABEL_ALL.value:
            logging.info("image model with densenet model (last) with multi-label classification")
            checkpoint = torch.load('experiments/results/classification_finetune.pth.tar')
            image_model.load_state_dict(checkpoint['model'])
        else:  # pretrained densenet model of ImageNet
            if enable_fine_tuning == False:
                raise Exception(
                    "To extract attr, the densenet should be already fine_tuned (as above) or requires fine-tune after pretraning of Imagenet")
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


class FutureAndAttrEncoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, model_type, encoded_image_size=14, enable_fine_tuning=False):
        super(FutureAndAttrEncoder, self).__init__()

        self.enc_image_size = encoded_image_size

        self.extractor, self.encoder_dim = get_image_extractor(model_type, enable_fine_tuning)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size))

        # Disable calculation of all gradients

        for p in self.extractor.image_model.parameters():
            p.requires_grad = False

        # Enable calculation of some gradients for fine tuning
        self.fine_tune(enable_fine_tuning)

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        features, attrs = self.extractor(
            images)  # (batch_size, 2048, image_size/32, image_size/32)

        # (batch_size, 2048, encoded_image_size, encoded_image_size)
        features = self.adaptive_pool(features)
        # (batch_size, encoded_image_size, encoded_image_size, 2048)
        # (later on the intermidiate dims are flatten: (prepare_inputs)
        # (batch_size, encoded_image_size*encoded_image_size, 2048)
        features = features.permute(0, 2, 3, 1)

        print("this is out.size2", features.size())
        print("this is attrs.size2", attrs.size())

        return features, attrs

    def fine_tune(self, enable_fine_tuning):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.extractor.image_model.children()):  # [5:]:#toda!!!
            for p in c.parameters():
                p.requires_grad = enable_fine_tuning


class FeaturesAndAttrAttention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(FeaturesAndAttrAttention, self).__init__()
        # linear layer to transform encoded image
        self.encoder_att = nn.Linear(encoder_dim + 512, attention_dim)
        # linear layer to transform decoder's output
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        # linear layer to calculate values to be softmax-ed
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_features, encoder_attr, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """

        encoder_attr = encoder_attr.unsqueeze(1).repeat(1, encoder_features.size(1), 1)
        att1 = self.encoder_att(torch.cat([encoder_features, encoder_attr], -1)
                                )  # (batch_size, l_regions, attention_dim)

        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)

        # (batch_size, num_pixels,1) -> com squeeze(2) fica (batch_size, l_regions)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)  # (batch_size, l_regions)
        attention_weighted_encoding = (
            encoder_features * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class ContinuousAttrAttentionDecoder(ContinuousDecoderWithAttentionAndImage):
    """
    Decoder.
    """

    def __init__(
            self, attention_dim, embedding_type, embed_dim, decoder_dim, vocab_size, token_to_id, encoder_dim=2048,
            dropout=0.5):

        super(ContinuousAttrAttentionDecoder, self).__init__(attention_dim, embedding_type,
                                                             embed_dim, decoder_dim, vocab_size, token_to_id, encoder_dim, dropout)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)

        self.attention = FeaturesAndAttrAttention(
            encoder_dim, decoder_dim, attention_dim)  # attention network

    def forward(self, word, encoder_features, encoder_attrs,  decoder_hidden_state, decoder_cell_state):
        attention_weighted_encoding, alpha = self.attention(encoder_features, encoder_attrs, decoder_hidden_state)
        embeddings = self.embedding(word)

        decoder_input = torch.cat((embeddings, attention_weighted_encoding), dim=1)

        decoder_hidden_state, decoder_cell_state = self.decode_step(
            decoder_input, (decoder_hidden_state, decoder_cell_state)
        )

        scores = self.fc(self.dropout(decoder_hidden_state))

        return scores, decoder_hidden_state, decoder_cell_state, alpha


class ContinuousAttentionAttrSoftmaxImageModel(ContinuousAttentionImageModel):

    def __init__(self,
                 args,
                 vocab_size,
                 token_to_id,
                 id_to_token,
                 max_len,
                 device
                 ):
        super().__init__(args, vocab_size, token_to_id, id_to_token, max_len, device)

    def _initialize_encoder_and_decoder(self):

        if (self.args.embedding_type not in [embedding.value for embedding in EmbeddingsType]):
            raise ValueError(
                "Continuous model should use pretrained embeddings...")

        self.encoder = FutureAndAttrEncoder(self.args.image_model_type,
                                            enable_fine_tuning=self.args.fine_tune_encoder)
        self.decoder = ContinuousAttrAttentionDecoder(
            encoder_dim=self.encoder.encoder_dim,
            attention_dim=self.args.attention_dim,
            decoder_dim=self.args.decoder_dim,
            embedding_type=self.args.embedding_type,
            embed_dim=self.args.embed_dim,
            vocab_size=self.vocab_size,
            token_to_id=self.token_to_id,
            dropout=self.args.dropout
        )

        self.decoder.normalize_embeddings()

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

    def _prepare_inputs_to_forward_pass(self, imgs, caps, caption_lengths):
        imgs = imgs.to(self.device)
        caps = caps.to(self.device)
        caption_lengths = caption_lengths.to(self.device)

        # encoder #TODO: MUDAR
        encoder_features, encoder_attrs = self.encoder(imgs)
        print("this is encoder out", encoder_features.size())
        encoder_features = encoder_features.view(
            encoder_features.size(0), -1, encoder_features.size(-1))  # flatten
        print("this is encoder flater", encoder_features.size())

        # sorted captions
        caption_lengths, sort_ind = caption_lengths.squeeze(
            1).sort(dim=0, descending=True)
        encoder_features = encoder_features[sort_ind]
        encoder_attrs = encoder_attrs[sort_ind]
        caps_sorted = caps[sort_ind]

        # input captions must not have "end_token"
        caption_lengths = (caption_lengths - 1).tolist()

        encoder_out = encoder_features, encoder_attrs

        return encoder_out, caps_sorted, caption_lengths

    def _predict(self, encoder_out, caps, caption_lengths):

        encoder_features, encoder_attrs = encoder_out

        batch_size = encoder_features.size(0)
        num_pixels = encoder_features.size(1)

        # Create tensors to hold word predicion scores and alphas
        all_predictions = torch.zeros(batch_size,  max(
            caption_lengths), self.decoder.embed_dim).to(self.device)
        all_alphas = torch.zeros(batch_size, max(
            caption_lengths), num_pixels).to(self.device)

        h, c = self.decoder.init_hidden_state(encoder_features)

        # Predict
        for t in range(max(
                caption_lengths)):
            # batchsizes of current time_step are the ones with lenght bigger than time-step (i.e have not fineshed yet)
            batch_size_t = sum([l > t for l in caption_lengths])

            predictions, h, c, alpha = self.decoder(
                caps[: batch_size_t, t],
                encoder_features[: batch_size_t],
                encoder_attrs[: batch_size_t],
                h[: batch_size_t],
                c[: batch_size_t])

            all_predictions[:batch_size_t, t, :] = predictions
            all_alphas[:batch_size_t, t, :] = alpha

        return {"predictions": all_predictions, "alphas": all_alphas}
