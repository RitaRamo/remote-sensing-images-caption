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


class VocabAttention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, vocab_dim, decoder_dim, embedding_vocab):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(VocabAttention, self).__init__()

        # linear layer to transform decoder's output
        self.decoder_att = nn.Linear(decoder_dim, vocab_dim)

        self.full_att = nn.Linear(vocab_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

        self.embedding_vocab = embedding_vocab

    def forward(self, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        # (batch_size, l_regions (512), regions_dim (300))
        vocab = self.embedding_vocab.repeat(decoder_hidden.size()[0], 1, 1)
        query = self.decoder_att(decoder_hidden)  # (batch_size, 1, encoder_dim)

        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        # (batch_size, num_pixels,1) -> com squeeze(2) fica (batch_size, l_regions)
        att = self.full_att(self.relu(vocab + query.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)  # (batch_size, l_regions)
        attention_weighted_encoding = (
            vocab * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class ContinuousDecoderWithAttentionOut(DecoderWithAttention):
    """
    Decoder.
    """

    def __init__(
            self, attention_dim, embedding_type, embed_dim, decoder_dim, vocab_size, token_to_id, post_processing,
            device, encoder_dim=2048, dropout=0.5):

        super(ContinuousDecoderWithAttentionOut, self).__init__(attention_dim, embedding_type,
                                                                embed_dim, decoder_dim, vocab_size, token_to_id, post_processing, encoder_dim, dropout)

        list_wordid = list(range(vocab_size))  # ignore first 4 special tokens : "start,end, unknow, padding"

        vocab = torch.transpose(torch.tensor(list_wordid).unsqueeze(-1), 0, 1)
        embedding_vocab = self.embedding(vocab).to(device)

        self.attention_out = VocabAttention(embed_dim, decoder_dim, embedding_vocab)  # attention network

        # replace softmax layer with embedding layer
        # self.fc = nn.Linear(decoder_dim, embed_dim)

    def forward(self, word, encoder_out, decoder_hidden_state, decoder_cell_state):
        attention_weighted_encoding, alpha = self.attention(encoder_out, decoder_hidden_state)
        embeddings = self.embedding(word)

        decoder_input = torch.cat((embeddings, attention_weighted_encoding), dim=1)

        decoder_hidden_state, decoder_cell_state = self.decode_step(
            decoder_input, (decoder_hidden_state, decoder_cell_state)
        )

        scores, alpha_out = self.attention_out(self.dropout(decoder_hidden_state))

        return scores, decoder_hidden_state, decoder_cell_state, alpha, alpha_out


class ContinuousAttentionOutModel(ContinuousAttentionModel):

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

        self.decoder = ContinuousDecoderWithAttentionOut(
            encoder_dim=self.encoder.encoder_dim,
            attention_dim=self.args.attention_dim,
            decoder_dim=self.args.decoder_dim,
            embedding_type=self.args.embedding_type,
            embed_dim=self.args.embed_dim,
            vocab_size=self.vocab_size,
            token_to_id=self.token_to_id,
            post_processing=self.args.post_processing,
            device=self.device,
            dropout=self.args.dropout
        )

        self.decoder.normalize_embeddings(self.args.no_normalization)

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

    def _predict(self, encoder_out, caps, caption_lengths):
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)

        # Create tensors to hold word predicion scores and alphas
        all_predictions = torch.zeros(batch_size,  max(
            caption_lengths), self.decoder.embed_dim).to(self.device)
        all_alphas = torch.zeros(batch_size, max(
            caption_lengths), num_pixels).to(self.device)
        all_alphas_out = torch.zeros(batch_size, max(
            caption_lengths), self.vocab_size).to(self.device)

        h, c = self.decoder.init_hidden_state(encoder_out)

        # Predict
        for t in range(max(
                caption_lengths)):
            # batchsizes of current time_step are the ones with lenght bigger than time-step (i.e have not fineshed yet)
            batch_size_t = sum([l > t for l in caption_lengths])

            predictions, h, c, alpha, alpha_out = self.decoder(
                caps[:batch_size_t, t], encoder_out[:batch_size_t], h[:batch_size_t], c[:batch_size_t])

            all_predictions[:batch_size_t, t, :] = predictions
            all_alphas[:batch_size_t, t, :] = alpha
            all_alphas_out[:batch_size_t, t, :] = alpha_out

        return {"predictions": all_predictions, "alphas": all_alphas, "alpha_out": all_alphas_out}

    def generate_output_index(self, input_word, encoder_out, h, c):
        predictions, h, c, _, _ = self.decoder(
            input_word, encoder_out, h, c)

        current_output_index = self._convert_prediction_to_output(predictions)

        return current_output_index, h, c
