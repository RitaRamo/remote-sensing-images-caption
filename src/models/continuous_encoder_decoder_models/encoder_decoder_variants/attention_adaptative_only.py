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
from data_preprocessing.preprocess_tokens import OOV_TOKEN
from embeddings.embeddings import EmbeddingsType
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention import ContinuousAttentionModel
from embeddings.embeddings import EmbeddingsType
from data_preprocessing.preprocess_tokens import START_TOKEN, END_TOKEN


class AdaptativeAttention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(AdaptativeAttention, self).__init__()
        # linear layer to transform encoded image
        self.linear_v = nn.Linear(decoder_dim, attention_dim)
        # linear layer to transform decoder's output
        self.linear_h = nn.Linear(decoder_dim, attention_dim)

        self.linear_s = nn.Linear(decoder_dim, attention_dim)

        # linear layer to calculate values to be softmax-ed
        self.linear_att_v = nn.Linear(attention_dim, 1)
        self.linear_att_s = nn.Linear(attention_dim, 1)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights
        self.dropout = nn.Dropout(0.5)

    def forward(self, encoder_out, decoder_hidden, st):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """

        #TODO: DROPOUT
        hidden = self.linear_h(decoder_hidden)  # (batch_size, attention_dim)
        # eq 6 - 8 (attention to image regions)
        v = self.linear_v(encoder_out)
        z_t = self.linear_att_v(self.tanh(v + hidden.unsqueeze(1))).squeeze(2)  # eq 6
        alpha_t = self.softmax(z_t)  # eq7
        c_t = (encoder_out * alpha_t.unsqueeze(2)).sum(dim=1)  # eq8

        # eq 12
        s = self.linear_s(st.unsqueeze(1))
        s_att = self.linear_att_s(self.tanh(s + hidden.unsqueeze(1))
                                  ).squeeze(2)  # right part of eq12
        z_t_extended = torch.cat((z_t, s_att), dim=1)
        alpha_t_hat = self.softmax(z_t_extended)  # eq12 completed

        beta_t = alpha_t_hat[:, -1].unsqueeze(1)
        c_hat_t = beta_t * st + (1 - beta_t) * c_t

        return c_hat_t, alpha_t, beta_t


class ContinuousDecoderWithAdaptativeAttention(DecoderWithAttention):
    """
    Decoder.
    """

    def __init__(
            self, attention_dim, embedding_type, embed_dim, decoder_dim, vocab_size, token_to_id, post_processing,
            encoder_dim=2048, dropout=0.5):

        super(ContinuousDecoderWithAdaptativeAttention, self).__init__(attention_dim, embedding_type,
                                                                       embed_dim, decoder_dim, vocab_size, token_to_id, post_processing, encoder_dim, dropout)

        # TODO: POR ATTENTION
        # replace softmax layer with embedding layer
        self.attention = AdaptativeAttention(
            decoder_dim, attention_dim)  # attention network

        self.fc = nn.Linear(decoder_dim, embed_dim)

        self.relu = nn.ReLU()
        self.linear_image = nn.Linear(encoder_dim, decoder_dim)
        self.linear_regions = nn.Linear(encoder_dim, decoder_dim)

        self.linear_dec = nn.Linear(decoder_dim, decoder_dim)  # TODO: chec
        self.linear_x = nn.Linear(embed_dim + decoder_dim, decoder_dim)

        self.decode_step = nn.LSTMCell(
            embed_dim, decoder_dim, bias=True)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        # before: (batch_size, encoded_image_size*encoded_image_size, 2048)
        mean_encoder_out = encoder_out.mean(dim=1)
        # (batch_size, 2048)
        # transform 2048 (dim image embeddings) in decoder dim
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        # c = self.init_c(mean_encoder_out)

        # transform V in same size as decoder to then use it in attention to add with sentinel
        global_image = self.linear_image(mean_encoder_out)  # (batch_size, 1, decoder_dim)
        V_spatial_features = self.linear_regions(encoder_out)  # (batch_size, image_size*image_size, decoder_dim)

        return h, h, global_image, V_spatial_features

    def forward(self, word, global_image, encoder_out, decoder_hidden_state, decoder_cell_state):
        embeddings = self.embedding(word)

        #(batch, embedding_dim + global_image_dim)
        inputs = torch.cat((embeddings, global_image), dim=-1)

        #(batch, decoder_dim)
        gt = self.sigmoid(self.linear_x(inputs) + self.linear_dec(decoder_hidden_state))

        #(batch, decoder_dim)
        decoder_hidden_state, decoder_cell_state = self.decode_step(
            embeddings, (decoder_hidden_state, decoder_cell_state)
        )

        #(batch, decoder_dim)
        st = gt * self.tanh(decoder_cell_state)

        # (batch, decoder_dim) [both visual features and st have d_dimenion (decoder dim)]
        adaptative_context_vector, alpha, beta = self.attention(
            encoder_out, decoder_hidden_state, st)  # attention of image

        #(batch, vocab/emb_dim)
        scores = self.fc(self.dropout(adaptative_context_vector + decoder_hidden_state))

        return scores, decoder_hidden_state, decoder_cell_state, alpha, beta


class ContinuousAdaptativeAttentionOnlyImageModel(ContinuousAttentionModel):

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

        self.decoder = ContinuousDecoderWithAdaptativeAttention(
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

    def _predict(self, encoder_out, caps, caption_lengths):
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)

        # Create tensors to hold word predicion scores and alphas
        all_predictions = torch.zeros(batch_size, max(
            caption_lengths), self.decoder.embed_dim).to(self.device)
        all_alphas = torch.zeros(batch_size, max(
            caption_lengths), num_pixels).to(self.device)
        all_betas = torch.zeros(batch_size, max(
            caption_lengths), 1).to(self.device)

        h, c, global_image, V_spatial_features = self.decoder.init_hidden_state(encoder_out)

        # Predict
        for t in range(max(
                caption_lengths)):
            # batchsizes of current time_step are the ones with lenght bigger than time-step (i.e have not fineshed yet)
            batch_size_t = sum([l > t for l in caption_lengths])

            predictions, h, c, alpha, beta = self.decoder(
                caps[: batch_size_t, t],
                global_image[: batch_size_t],
                V_spatial_features[: batch_size_t],
                h[: batch_size_t],
                c[: batch_size_t])

            all_predictions[:batch_size_t, t, :] = predictions
            all_alphas[:batch_size_t, t, :] = alpha
            all_betas[:batch_size_t, t, :] = beta

        return {"predictions": all_predictions, "alphas": all_alphas, "betas": all_betas}
