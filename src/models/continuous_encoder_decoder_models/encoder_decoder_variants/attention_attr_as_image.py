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
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention import ContinuousAttentionModel, ContinuousDecoderWithAttention
from embeddings.embeddings import EmbeddingsType
from preprocess_data.images import ImageNetModelsPretrained
import logging
from torchvision import models
from preprocess_data.tokens import START_TOKEN, END_TOKEN
from preprocess_data.images import get_image_extractor, DenseNetFeatureAndAttrExtractor

# chamar image models


# def get_image_extractor(model_type, enable_fine_tuning):
#     if model_type == ImageNetModelsPretrained.DENSENET.value or model_type == ImageNetModelsPretrained.MULTILABEL_ALL.value:
#         logging.info("image model with densenet model")
#         vocab_size = 512

#         image_model = models.densenet201(pretrained=True)
#         image_model.classifier = nn.Linear(image_model.classifier.in_features, vocab_size)

#         encoder_dim = image_model.classifier.in_features

#         if model_type == ImageNetModelsPretrained.MULTILABEL_ALL.value:
#             logging.info("image model with densenet model (all) with multi-label classification")
#             checkpoint = torch.load('experiments/results/classification_densenet_modifiedrsicd.pth.tar')
#             image_model.load_state_dict(checkpoint['model'])
#         else:  # pretrained densenet model of ImageNet
#             if enable_fine_tuning == False:
#                 raise Exception(
#                     "To extract attr, the densenet should be already fine_tuned (as above) or requires fine-tune after pretraning of Imagenet")
#         return DenseNetFeatureAndAttrExtractor(image_model), encoder_dim
#     else:
#         raise Exception("not implemented extractor for the other types")


# class DenseNetFeatureAndAttrExtractor(nn.Module):
#     def __init__(self, image_model):
#         super().__init__()
#         self.image_model = image_model

#     def forward(self, x):
#         modules_features = list(self.image_model.children())[:-1]
#         features_extractor = nn.Sequential(*modules_features)

#         features = features_extractor(x)
#         attrs = self.image_model(x)

#         return features, attrs


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


class ContinuousDecoderWithAttentionAsAttrImage(ContinuousDecoderWithAttention):
    """
    Decoder.
    """

    def __init__(
            self, attention_dim, embedding_type, embed_dim, decoder_dim, vocab_size, token_to_id, post_processing,
            device, encoder_dim=2048, dropout=0.5):

        super(ContinuousDecoderWithAttentionAsAttrImage, self).__init__(attention_dim, embedding_type,
                                                                        embed_dim, decoder_dim, vocab_size, token_to_id, post_processing, encoder_dim, dropout)

        classification_state = torch.load("src/data/RSICD/datasets/classification_dataset")
        list_wordid = classification_state["list_wordid"]

        encoder_attrs_classes = torch.transpose(torch.tensor(list_wordid).unsqueeze(-1), 0, 1)
        self.embedding_attr = self.embedding(encoder_attrs_classes).to(device)

        self.image_embedding = None

    def init_hidden_state(self, encoder_features, encoder_attr):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_features = encoder_features.mean(dim=1)
        h = self.init_h(mean_encoder_features)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_features)

        batch_attr_embedding = self.embedding_attr.repeat(encoder_attr.size()[0], 1, 1)

        mean_attr = torch.sum(batch_attr_embedding*encoder_attr.unsqueeze(2), dim=1)/torch.sum(encoder_attr)

        self.image_embedding = mean_attr

        return h, c


class ContinuousAttentionAttrAsImageModel(ContinuousAttentionModel):

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
        self.decoder = ContinuousDecoderWithAttentionAsAttrImage(
            encoder_dim=self.encoder.encoder_dim,
            attention_dim=self.args.attention_dim,
            decoder_dim=self.args.decoder_dim,
            embedding_type=self.args.embedding_type,
            embed_dim=self.args.embed_dim,
            vocab_size=self.vocab_size,
            token_to_id=self.token_to_id,
            post_processing=self.args.post_processing,
            dropout=self.args.dropout,
            device=self.device
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
        encoder_features = encoder_features.view(
            encoder_features.size(0), -1, encoder_features.size(-1))  # flatten

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

        h, c = self.decoder.init_hidden_state(encoder_features, encoder_attrs)

        # Predict
        for t in range(max(
                caption_lengths)):
            # batchsizes of current time_step are the ones with lenght bigger than time-step (i.e have not fineshed yet)
            batch_size_t = sum([l > t for l in caption_lengths])

            predictions, h, c, alpha = self.decoder(
                caps[: batch_size_t, t],
                encoder_features[: batch_size_t],
                h[: batch_size_t],
                c[: batch_size_t])

            all_predictions[:batch_size_t, t, :] = predictions
            all_alphas[:batch_size_t, t, :] = alpha

        return {"predictions": all_predictions, "alphas": all_alphas}

    def inference_with_greedy(self, image, n_solutions=0):
        with torch.no_grad():  # no need to track history

            decoder_sentence = []

            input_word = torch.tensor([self.token_to_id[START_TOKEN]])

            i = 1

            encoder_features, encoder_attrs = self.encoder(image)
            encoder_features = encoder_features.view(encoder_features.size(0), -1, encoder_features.size(-1))  # flatten

            h, c = self.decoder.init_hidden_state(encoder_features, encoder_attrs)

            while True:

                scores, h, c = self.generate_output_index(
                    input_word, encoder_features, h, c)

                sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)

                current_output_index = sorted_indices[0]

                current_output_token = self.id_to_token[current_output_index.item(
                )]

                decoder_sentence.append(current_output_token)

                if current_output_token == END_TOKEN:
                    # ignore end_token
                    decoder_sentence = decoder_sentence[:-1]
                    break

                if i >= self.max_len-1:  # until 35
                    break

                input_word[0] = current_output_index.item()

                i += 1

            generated_sentence = " ".join(decoder_sentence)
            print("\ngenerated sentence:", generated_sentence)

            return generated_sentence  # input_caption

    def generate_output_index(self, input_word, encoder_features,  h, c):
        predictions, h, c, _ = self.decoder(
            input_word, encoder_features,  h, c)

        current_output_index = self._convert_prediction_to_output(predictions)

        return current_output_index, h, c
