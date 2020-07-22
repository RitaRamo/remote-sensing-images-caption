import torchvision
from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from models.basic_encoder_decoder_models.encoder_decoder import Encoder, Decoder
from models.abtract_model import AbstractEncoderDecoderModel
import torch.nn.functional as F
from embeddings.embeddings import get_embedding_layer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from data_preprocessing.preprocess_tokens import OOV_TOKEN
from embeddings.embeddings import EmbeddingsType
from models.continuous_encoder_decoder_models.encoder_decoder import ContinuousEncoderDecoderModel
from embeddings.embeddings import EmbeddingsType
from utils.enums import ContinuousLossesType


class ContinuousDecoderWithImage(Decoder):

    def __init__(self, decoder_dim, embed_dim, embedding_type, vocab_size, token_to_id, post_processing,
                 encoder_dim=2048, dropout=0.5):

        super(ContinuousDecoderWithImage, self).__init__(decoder_dim, embed_dim,
                                                         embedding_type, vocab_size, token_to_id, post_processing, encoder_dim, dropout)

        # linear layer to find representation of image
        self.represent_image = nn.Linear(encoder_dim, embed_dim)
        self.image_embedding = None

        # replace softmax with a embedding layer
        self.fc = nn.Linear(decoder_dim, embed_dim)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)

        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        self.image_embedding = self.represent_image(mean_encoder_out)

        return h, h


class ContinuousEncoderDecoderImageWModel(ContinuousEncoderDecoderModel):

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

    def _calculate_loss(self, predict_output, caps, caption_lengths):
        predictions = predict_output["predictions"]
        targets = caps[:, 1:]  # targets doesnt have stark token

        target_embeddings = self.decoder.embedding(targets).to(self.device)

        if self.args.no_normalization == False:
            # when target embeddings start normalized, predictions should also be normalized
            predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)

        loss = self.loss_method(
            predictions,
            target_embeddings,
            caption_lengths,
        )

        return loss

    def _define_loss_criteria(self):
        loss_type = self.args.continuous_loss_type

        if loss_type == ContinuousLossesType.COS_AVG_SENTENCE_AND_INPUTS_W.value:
            self.loss_method = self.cos_avg_sentence_and_inputs_w_loss
            self.criterion = nn.CosineEmbeddingLoss().to(self.device)
        else:
            raise Exception("only available: COS_AVG_SENTENCE_AND_INPUTS_W ")

    def cos_avg_sentence_and_inputs_w_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths
    ):
        word_losses = 0.0  # pred_against_target_loss; #pred_sentence_again_target_sentence;"pred_sentence_agains_image
        sentence_losses = 0.0
        input1_losses = 0.0
        input2_losses = 0.0

        images_embedding = self.decoder.image_embedding

        n_sentences = predictions.size()[0]
        for i in range(n_sentences):  # iterate by sentence
            preds_without_padd = predictions[i, :caption_lengths[i], :]
            targets_without_padd = target_embeddings[i, :caption_lengths[i], :]
            y = torch.ones(targets_without_padd.shape[0]).to(self.device)

            # word-level loss   (each prediction against each target)
            word_losses += self.criterion(
                preds_without_padd,
                targets_without_padd,
                y
            )

            # sentence-level loss (sentence predicted agains target sentence)
            sentence_mean_pred = torch.mean(preds_without_padd, dim=0).unsqueeze(0)  # ver a dim
            sentece_mean_target = torch.mean(targets_without_padd, dim=0).unsqueeze(0)

            y = torch.ones(1).to(self.device)

            sentence_losses += self.criterion(
                sentence_mean_pred,
                sentece_mean_target,
                y
            )

            image_embedding = images_embedding[i].unsqueeze(0)

            # 1º input loss (sentence predicted against input image)
            input1_losses += self.criterion(
                sentence_mean_pred,
                image_embedding,
                y
            )

            # 2º input loss (sentence predicted against input image)
            input2_losses += self.criterion(
                image_embedding,
                sentece_mean_target,
                y
            )

        word_loss = word_losses / n_sentences
        sentence_loss = sentence_losses / n_sentences
        input1_loss = input1_losses / n_sentences
        input2_loss = input2_losses / n_sentences

        loss = word_loss + self.args.w2 * sentence_loss + self.args.w3 * input1_loss + self.args.w4 * input2_loss

        return loss
