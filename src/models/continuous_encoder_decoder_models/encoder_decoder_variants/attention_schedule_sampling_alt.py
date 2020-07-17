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
from models.continuous_encoder_decoder_models.encoder_decoder import ContinuousEncoderDecoderModel
from embeddings.embeddings import EmbeddingsType


class ContinuousDecoderWithAttentionAndScheduleSampling(DecoderWithAttention):
    """
    Decoder.
    """

    def __init__(
            self, attention_dim, embedding_type, embed_dim, decoder_dim, vocab_size, token_to_id, encoder_dim=2048,
            dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(ContinuousDecoderWithAttentionAndScheduleSampling, self).__init__(attention_dim,
                                                                                embedding_type, embed_dim, decoder_dim, vocab_size, token_to_id, encoder_dim, dropout)
        # instead of being bla bla
        self.fc = nn.Linear(decoder_dim, embed_dim)

    def forward(
            self, sampling_rate, word_ground_truth, embedding_predicted, encoder_out, decoder_hidden_state,
            decoder_cell_state):

        attention_weighted_encoding, alpha = self.attention(encoder_out,
                                                            decoder_hidden_state)

        embedding_ground = self.embedding(word_ground_truth)

        embedding = (sampling_rate) * embedding_predicted + (1 - sampling_rate) * embedding_ground

        decoder_input = torch.cat(
            (embedding, attention_weighted_encoding), dim=1
        )

        decoder_hidden_state, decoder_cell_state = self.decode_step(
            decoder_input, (decoder_hidden_state, decoder_cell_state)
        )

        scores = self.fc(self.dropout(decoder_hidden_state))

        return scores, decoder_hidden_state, decoder_cell_state, alpha

    def inference(self, word_predicted, encoder_out, decoder_hidden_state, decoder_cell_state):

        attention_weighted_encoding, alpha = self.attention(encoder_out,
                                                            decoder_hidden_state)

        embeddings = self.embedding(word_predicted)

        decoder_input = torch.cat(
            (embeddings, attention_weighted_encoding), dim=1
        )

        decoder_hidden_state, decoder_cell_state = self.decode_step(
            decoder_input, (decoder_hidden_state, decoder_cell_state)
        )

        scores = self.fc(self.dropout(decoder_hidden_state))

        return scores, decoder_hidden_state, decoder_cell_state, alpha


class ContinuousAttentionWithScheduleSamplingAltModel(ContinuousEncoderDecoderModel):

    SAMPLING_INITIAL_RATE = 0.02

    def __init__(self,
                 args,
                 vocab_size,
                 token_to_id,
                 id_to_token,
                 max_len,
                 device
                 ):
        super().__init__(args, vocab_size, token_to_id, id_to_token, max_len, device)
        self.rate_step = 0

    def _initialize_encoder_and_decoder(self):

        if (self.args.embedding_type not in [embedding.value for embedding in EmbeddingsType]):
            raise ValueError(
                "Continuous model should use pretrained embeddings...")

        self.encoder = Encoder(self.args.image_model_type,
                               enable_fine_tuning=self.args.fine_tune_encoder)

        self.decoder = ContinuousDecoderWithAttentionAndScheduleSampling(
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

        # Create tensors to hold word predicion scores and alphas
        all_predictions = torch.zeros(batch_size, max(
            caption_lengths), self.decoder.embed_dim).to(self.device)
        all_alphas = torch.zeros(batch_size, max(
            caption_lengths), num_pixels).to(self.device)

        h, c = self.decoder.init_hidden_state(encoder_out)

        if self.current_epoch > 10:
            sampling_rate = self.SAMPLING_INITIAL_RATE * self.rate_step
        else:
            sampling_rate = 0.0
            self.rate_step = 0.0

        # 1º time-step  (embedding of ground-truth given as last prediction since there is no prediction yet)
        predictions, h, c, alpha = self.decoder(
            sampling_rate,
            caps[:, 0],
            self.decoder.embedding(caps[:, 0]),
            encoder_out, h, c)
        all_predictions[:, 0, :] = predictions
        all_alphas[:, 0, :] = alpha

        # remain time-steps
        for t in range(1, max(
                caption_lengths)):
            # batchsizes of current time_step are the ones with lenght bigger than time-step (i.e have not fineshed yet)
            batch_size_t = sum([l > t for l in caption_lengths])

            predictions, h, c, alpha = self.decoder(
                sampling_rate,
                caps[: batch_size_t, t],
                all_predictions[: batch_size_t, t - 1, :],
                encoder_out[: batch_size_t],
                h[: batch_size_t],
                c[: batch_size_t]
            )

            all_predictions[:batch_size_t, t, :] = predictions
            all_alphas[:batch_size_t, t, :] = alpha

        return {"predictions": all_predictions, "alphas": all_alphas}

    def generate_output_index(self, input_word, encoder_out, h, c):

        predictions, h, c, _ = self.decoder.inference(input_word, encoder_out, h, c)

        current_output_index = self._convert_prediction_to_output(predictions)

        return current_output_index, h, c

    def _save_checkpoint(self, val_loss_improved, epoch, epochs_since_last_improvement, val_loss):
        if val_loss_improved:

            state = {'epoch': epoch,
                     'epochs_since_last_improvement': epochs_since_last_improvement,
                     'val_loss': val_loss,
                     'encoder': self.encoder.state_dict(),
                     'decoder': self.decoder.state_dict(),
                     'encoder_optimizer': self.encoder_optimizer.state_dict() if self.encoder_optimizer else None,
                     'decoder_optimizer': self.decoder_optimizer.state_dict(),
                     'rate': self.rate_step
                     }

            if self.rate_step < 50:
                self.rate_step += 1

            torch.save(state, self.get_checkpoint_path())
