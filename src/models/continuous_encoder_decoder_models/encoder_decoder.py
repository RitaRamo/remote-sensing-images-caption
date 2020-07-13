import torchvision
from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from models.basic_encoder_decoder_models.encoder_decoder import Encoder, Decoder
import torch.nn.functional as F
from embeddings.embeddings import get_embedding_layer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from preprocess_data.tokens import OOV_TOKEN
from embeddings.embeddings import EmbeddingsType
from models.abtract_model import AbstractEncoderDecoderModel
from models.continuous_encoder_decoder_models.continuous_losses import ContinuousLossesType, ContinuousLoss


class ContinuousDecoder(Decoder):
    """
    Decoder.
    """

    def __init__(self, decoder_dim, embed_dim, embedding_type, vocab_size, token_to_id, post_processing,
                 encoder_dim=2048, dropout=0.5):

        super(ContinuousDecoder, self).__init__(decoder_dim, embed_dim, embedding_type,
                                                vocab_size, token_to_id, post_processing, encoder_dim, dropout)

        # replace softmax with a embedding layer
        self.fc = nn.Linear(decoder_dim, embed_dim)


class ContinuousEncoderDecoderModel(AbstractEncoderDecoderModel):

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

        self.decoder = ContinuousDecoder(
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

    def _define_loss_criteria(self):
        self.criteria = ContinuousLoss(
            self.args.continuous_loss_type, self.device, self.decoder)

    def _predict(self, encoder_out, caps, caption_lengths):
        batch_size = encoder_out.size(0)

        # Create tensors to hold word predicion scores
        all_predictions = torch.zeros(batch_size, max(
            caption_lengths), self.decoder.embed_dim).to(self.device)

        h, c = self.decoder.init_hidden_state(encoder_out)

        # Predict
        for t in range(max(caption_lengths)):
            # batchsizes of current time_step are the ones with lenght bigger than time-step (i.e have not fineshed yet)
            batch_size_t = sum([l > t for l in caption_lengths])

            predictions, h, c = self.decoder(
                caps[:batch_size_t, t], encoder_out[:batch_size_t], h[:batch_size_t], c[:batch_size_t])

            all_predictions[:batch_size_t, t, :] = predictions

        # return is a dict, since for other models additional things maybe be return: ex attention model-> add alphas
        return {"predictions": all_predictions}

    def _calculate_loss(self, predict_output, caps, caption_lengths):
        predictions = predict_output["predictions"]
        targets = caps[:, 1:]  # targets doesnt have stark token

        target_embeddings = self.decoder.embedding(targets).to(self.device)

        if self.args.no_normalization == False:
            # when target embeddings start normalized, predictions should also be normalized
            predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)

        loss = self.criteria.compute_loss(
            predictions,
            target_embeddings,
            caption_lengths,
        )

        return loss

    def generate_output_index(self, input_word, encoder_out, h, c):
        predictions, h, c = self.decoder(
            input_word, encoder_out, h, c)

        current_output_index = self._convert_prediction_to_output(predictions)

        return current_output_index, h, c

    def _convert_prediction_to_output(self, predictions):
        output = torch.cosine_similarity(
            self.decoder.embedding.weight.data, predictions)

        return output
