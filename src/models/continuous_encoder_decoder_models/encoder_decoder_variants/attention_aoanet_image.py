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
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_image import ContinuousAttentionImageModel
from embeddings.embeddings import EmbeddingsType


class ContinuousDecoderSoftWithAoANet(DecoderWithAttention):
    """
    Decoder.
    """

    def __init__(
            self, attention_dim, embedding_type, embed_dim, decoder_dim, vocab_size, token_to_id, encoder_dim=2048,
            dropout=0.5):

        super(ContinuousDecoderSoftWithAoANet, self).__init__(attention_dim, embedding_type,
                                                              embed_dim, decoder_dim, vocab_size, token_to_id, encoder_dim, dropout)

        # linear layer to find representation of image
        self.represent_image = nn.Linear(encoder_dim, embed_dim)
        self.image_embedding = None

        self.aoa_layer = nn.Sequential(nn.Linear(decoder_dim + encoder_dim, encoder_dim*2), nn.GLU())

        # replace softmax layer with embedding layer
        self.fc = nn.Linear(encoder_dim, embed_dim)

    def init_hidden_state(self, mean_encoder_out):

        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        self.image_embedding = self.represent_image(mean_encoder_out)

        return h, h

    def forward(self, word, encoder_out, decoder_hidden_state, decoder_cell_state, mean_encoder_out, context_vector):
        embeddings = self.embedding(word)

        # TODO: RETURN self.mean_encoder_out
        lstm_input = torch.cat([embeddings, mean_encoder_out + context_vector], 1)

        decoder_hidden_state, decoder_cell_state = self.decode_step(
            lstm_input, (decoder_hidden_state, decoder_hidden_state))

        attention_weighted_encoding, alpha = self.attention(encoder_out, decoder_hidden_state)

        h_with_att = torch.cat([attention_weighted_encoding, decoder_hidden_state], 1)

        context_vector = self.aoa_layer(h_with_att)

        scores = self.fc(self.dropout(context_vector))

        return scores, decoder_hidden_state, decoder_cell_state, context_vector, alpha


class ContinuousAttentionAoANetImageModel(ContinuousAttentionImageModel):

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

        self.decoder = ContinuousDecoderSoftWithAoANet(
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

    def _predict(self, encoder_out, caps, caption_lengths):
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)
        encoder_dim = encoder_out.size(2)

        # Create tensors to hold word predicion scores and alphas
        all_predictions = torch.zeros(batch_size,  max(
            caption_lengths), self.decoder.embed_dim).to(self.device)
        all_alphas = torch.zeros(batch_size, max(
            caption_lengths), num_pixels).to(self.device)

        mean_encoder_out = encoder_out.mean(dim=1)

        h, c = self.decoder.init_hidden_state(mean_encoder_out)
        context_vector = torch.zeros(batch_size, encoder_dim).to(self.device)

        # Predict
        for t in range(max(
                caption_lengths)):
            # batchsizes of current time_step are the ones with lenght bigger than time-step (i.e have not fineshed yet)
            batch_size_t = sum([l > t for l in caption_lengths])

            predictions, h, c, context_vector, alpha = self.decoder(
                caps[: batch_size_t, t],
                encoder_out[: batch_size_t],
                h[: batch_size_t],
                c[: batch_size_t],
                mean_encoder_out[: batch_size_t],
                context_vector[: batch_size_t]
            )

            all_predictions[:batch_size_t, t, :] = predictions
            all_alphas[:batch_size_t, t, :] = alpha

        return {"predictions": all_predictions, "alphas": all_alphas}
