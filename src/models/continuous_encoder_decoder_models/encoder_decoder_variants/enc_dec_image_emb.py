import torchvision
from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from models.basic_encoder_decoder_models.encoder_decoder import Decoder
from models.abtract_model import AbstractEncoderDecoderModel
import torch.nn.functional as F
from embeddings.embeddings import get_embedding_layer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from data_preprocessing.preprocess_tokens import OOV_TOKEN
from embeddings.embeddings import EmbeddingsType
from models.continuous_encoder_decoder_models.encoder_decoder import ContinuousEncoderDecoderModel
from embeddings.embeddings import EmbeddingsType
from data_preprocessing.preprocess_images import get_image_model


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, model_type, encoded_image_size=14, enable_fine_tuning=False):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        self.model, self.encoder_dim = get_image_model(model_type)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size))

        # Disable calculation of all gradients
        for p in self.model.parameters():
            p.requires_grad = False

        # Enable calculation of some gradients for fine tuning
        self.fine_tune(enable_fine_tuning)

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        # out = self.model(
        #     images)  # (batch_size, 2048, image_size/32, image_size/32)

        out = self.model.extract_features(images)
        # #print("image size", out.size())

        # # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = self.adaptive_pool(out)
        # # (batch_size, encoded_image_size, encoded_image_size, 2048)
        # # (later on the intermidiate dims are flatten: (prepare_inputs)
        # # (batch_size, encoded_image_size*encoded_image_size, 2048)
        out = out.permute(0, 2, 3, 1)

        return out

    def fine_tune(self, enable_fine_tuning):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.model.children())[-1:]:  # [5:]:#toda!!!
            print("Lets fine tune this childreen:", c)
            for p in c.parameters():
                p.requires_grad = enable_fine_tuning


class ContinuousDecoderWithImage(Decoder):

    def __init__(self, decoder_dim, embed_dim, embedding_type, vocab_size, token_to_id, post_processing,
                 encoder_dim=2048, dropout=0.5):

        super(ContinuousDecoderWithImage, self).__init__(decoder_dim, embed_dim,
                                                         embedding_type, vocab_size, token_to_id, post_processing, encoder_dim, dropout)

        self.image_embedding = None

        # replace softmax with a embedding layer
        self.fc = nn.Linear(decoder_dim, embed_dim)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)

        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        self.image_embedding = mean_encoder_out

        return h, h


class ContinuousEncoderDecoderImageEmbModel(ContinuousEncoderDecoderModel):

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

        self.encoder = Encoder(self.args.image_model_type,
                               enable_fine_tuning=self.args.fine_tune_encoder)

        self.decoder = ContinuousDecoderWithImage(
            encoder_dim=self.encoder.encoder_dim,
            decoder_dim=self.args.decoder_dim,
            embedding_type=self.args.embedding_type,
            embed_dim=self.args.embed_dim,
            vocab_size=self.vocab_size,
            token_to_id=self.token_to_id,
            post_processing=self.args.post_processing,
            dropout=self.args.dropout
        )

        self.decoder.normalize_embeddings(self.args.no_normalization)

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
