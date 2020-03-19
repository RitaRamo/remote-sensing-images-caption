import torchvision
from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from models.basic_encoder_decoder_models.encoder_decoder import Encoder
from models.basic_encoder_decoder_models.encoder_decoder_variants.attention import BasicAttentionModel, Attention, DecoderWithAttention
import torch.nn.functional as F
from embeddings.embeddings import get_embedding_layer


class SATDecoder(DecoderWithAttention):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embedding_type, embed_dim, decoder_dim, vocab_size, token_to_id, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(SATDecoder, self).__init__(attention_dim, embedding_type,
                                         embed_dim, decoder_dim, vocab_size, token_to_id, encoder_dim, dropout)

       # linear layer to create a sigmoid-activated gate
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, word, encoder_out, decoder_hidden_state, decoder_cell_state):

        attention_weighted_encoding, alpha = self.attention(encoder_out,
                                                            decoder_hidden_state)

        # gating scalar, (batch_size_t, encoder_dim)
        gate = self.sigmoid(self.f_beta(decoder_hidden_state))
        attention_weighted_encoding = gate * attention_weighted_encoding

        embeddings = self.embedding(word)

        decoder_input = torch.cat(
            (embeddings, attention_weighted_encoding), dim=1
        )

        decoder_hidden_state, decoder_cell_state = self.decode_step(
            decoder_input, (decoder_hidden_state, decoder_cell_state)
        )

        scores = self.fc(self.dropout(decoder_hidden_state))

        return scores, decoder_hidden_state, decoder_cell_state, alpha


class BasicShowAttendAndTellModel(BasicAttentionModel):

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

        self.encoder = Encoder(self.args.image_model_type,
                               enable_fine_tuning=self.args.fine_tune_encoder)

        self.decoder = SATDecoder(
            encoder_dim=self.encoder.encoder_dim,
            attention_dim=self.args.attention_dim,
            decoder_dim=self.args.decoder_dim,
            embedding_type=self.args.embedding_type,
            embed_dim=self.args.embed_dim,
            vocab_size=self.vocab_size,
            token_to_id=self.token_to_id,
            dropout=self.args.dropout
        )

        self.encoder = self.encoder.to(self.device)

        self.decoder = self.decoder.to(self.device)
