import torchvision
from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from models.basic_encoder_decoder_models.encoder_decoder_variants.attention import Attention, Encoder, DecoderWithAttention
from models.abtract_model import AbstractEncoderDecoderModel
import torch.nn.functional as F
from embeddings.embeddings import get_embedding_layer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from preprocess_data.tokens import OOV_TOKEN
from embeddings.embeddings import EmbeddingsType
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention import ContinuousAttentionModel
from embeddings.embeddings import EmbeddingsType
from preprocess_data.tokens import START_TOKEN, END_TOKEN
import math


class ProductAttention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(ProductAttention, self).__init__()
        # linear layer to transform decoder's output
        self.decoder_att = nn.Linear(decoder_dim, encoder_dim)
        self.dk = encoder_dim
        # linear layer to calculate values to be softmax-ed
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        query = self.decoder_att(decoder_hidden).unsqueeze(1)  # (batch_size, l_regions,1)

        #scores = (batch_size,1,l_regions)
        scores = torch.matmul(query, encoder_out.transpose(-2, -1)) \
            / math.sqrt(self.dk)

        #scores = (batch_size,l_regions)
        scores = scores.squeeze(1)
        alpha = self.softmax(scores)  # (batch_size, l_regions)
        attention_weighted_encoding = (
            encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class ContinuousDecoderWithProductAttentionAndImage(DecoderWithAttention):
    """
    Decoder.
    """

    def __init__(
            self, attention_dim, embedding_type, embed_dim, decoder_dim, vocab_size, token_to_id, post_processing,
            encoder_dim=2048, dropout=0.5):

        super(ContinuousDecoderWithProductAttentionAndImage, self).__init__(attention_dim, embedding_type,
                                                                            embed_dim, decoder_dim, vocab_size, token_to_id, post_processing, encoder_dim, dropout)

        # linear layer to find representation of image
        self.represent_image = nn.Linear(decoder_dim, embed_dim)
        self.image_embedding = None

        # replace softmax layer with embedding layer
        self.fc = nn.Linear(decoder_dim, embed_dim)

        self.attention = ProductAttention(encoder_dim, decoder_dim)  # attention network

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)

        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim) 512 units

        self.image_embedding = self.represent_image(h)  # 300

        return h, h


class ContinuousProductAttentionImageAltModel(ContinuousAttentionModel):

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

        self.decoder = ContinuousDecoderWithProductAttentionAndImage(
            encoder_dim=self.encoder.encoder_dim,
            attention_dim=self.args.attention_dim,
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
