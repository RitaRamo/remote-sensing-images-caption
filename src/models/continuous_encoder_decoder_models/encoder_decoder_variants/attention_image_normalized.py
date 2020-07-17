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
from utils.enums import ContinuousLossesType


class ContinuousDecoderWithAttentionAndImage(DecoderWithAttention):
    """
    Decoder.
    """

    def __init__(
            self, attention_dim, embedding_type, embed_dim, decoder_dim, vocab_size, token_to_id, post_processing,
            encoder_dim=2048, dropout=0.5):

        super(ContinuousDecoderWithAttentionAndImage, self).__init__(attention_dim, embedding_type,
                                                                     embed_dim, decoder_dim, vocab_size, token_to_id, post_processing, encoder_dim, dropout)

        # linear layer to find representation of image
        self.represent_image = nn.Linear(encoder_dim, embed_dim)
        self.image_embedding = None

        # replace softmax layer with embedding layer
        self.fc = nn.Linear(decoder_dim, embed_dim)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)

        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim) 512 units

        self.image_embedding = self.represent_image(mean_encoder_out)  # 300 512

        return h, h

    def inference(self, embeddings, encoder_out, decoder_hidden_state, decoder_cell_state):
        attention_weighted_encoding, alpha = self.attention(encoder_out, decoder_hidden_state)

        decoder_input = torch.cat((embeddings, attention_weighted_encoding), dim=1)

        decoder_hidden_state, decoder_cell_state = self.decode_step(
            decoder_input, (decoder_hidden_state, decoder_cell_state)
        )

        scores = self.fc(self.dropout(decoder_hidden_state))

        return scores, decoder_hidden_state, decoder_cell_state, alpha


class ContinuousAttentionImageNormalizedModel(ContinuousAttentionModel):

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

        self.decoder = ContinuousDecoderWithAttentionAndImage(
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

    def generate_output_embedding(self, input_embedding, encoder_out, h, c):
        predictions, h, c, _ = self.decoder.inference(
            input_embedding, encoder_out, h, c)

        current_output_index = self._convert_prediction_to_output(predictions)

        return predictions, current_output_index, h, c

    def _define_loss_criteria(self):
        loss_type = self.args.continuous_loss_type

        if loss_type == ContinuousLossesType.COSNONORM_13_NORM.value:
            self.loss_method = self.cos_nonorm_and_third_norm_with_normalizedemb_loss
            self.criterion = nn.CosineEmbeddingLoss().to(self.device)

        else:
            raise Exception("only available: COSNONORM_13_NORM ")

    def _calculate_loss(self, predict_output, caps, caption_lengths):
        predictions = predict_output["predictions"]
        targets = caps[:, 1:]  # targets doesnt have stark token

        target_embeddings = self.decoder.embedding(targets).to(self.device)

        loss = self.loss_method(
            predictions,
            target_embeddings,
            caption_lengths,
        )

        return loss

    def cos_nonorm_and_third_norm_with_normalizedemb_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths
    ):
        word_losses = 0.0  # pred_against_target_loss; #pred_sentence_again_target_sentence;"pred_sentence_agains_image
        input1_losses = 0.0

        images_embedding = self.decoder.image_embedding

        targets_batch_norms = target_embeddings.norm(p=2, dim=1)
        min_targets_batch_norms = torch.min(targets_batch_norms)
        max_targets_batch_norms = torch.max(targets_batch_norms)

        preds_batch_norms = predictions.norm(p=2, dim=1)
        min_preds_batch_norms = torch.min(preds_batch_norms)
        max_preds_batch_norms = torch.max(preds_batch_norms)

        param_a = 0.5
        param_b = 1.0

        n_sentences = predictions.size()[0]
        for i in range(n_sentences):  # iterate by sentence
            preds_without_padd = predictions[i, :caption_lengths[i], :]
            targets_without_padd = target_embeddings[i, :caption_lengths[i], :]
            y = torch.ones(targets_without_padd.shape[0]).to(self.device)

            word_losses += self.criterion(
                preds_without_padd,
                targets_without_padd,
                y
            )

            # sentence-level loss (sentence predicted agains target sentence)
            preds_norms = preds_without_padd.norm(p=2, dim=1)
            normalized_preds_norms = param_a + ((preds_norms - min_preds_batch_norms)
                                                * (param_b - param_a)) / (max_preds_batch_norms - min_preds_batch_norms)
            preds_without_padd = torch.nn.functional.normalize(preds_without_padd, p=2, dim=-1)

            sentence_mean_pred = torch.sum(preds_without_padd * normalized_preds_norms.unsqueeze(1),
                                           dim=0).unsqueeze(0) / torch.sum(normalized_preds_norms)  # ver a dim

            y = torch.ones(1).to(self.device)

            image_embedding = images_embedding[i].unsqueeze(0)

            # 1º input loss (sentence predicted against input image)
            input1_losses += self.criterion(
                sentence_mean_pred,
                image_embedding,
                y
            )

        word_loss = word_losses / n_sentences
        input1_loss = input1_losses / n_sentences

        loss = word_loss + input1_loss

        return loss
