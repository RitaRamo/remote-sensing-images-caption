import torchvision
from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from models.abtract_model import AbstractEncoderDecoderModel
import torch.nn.functional as F
from embeddings.embeddings import get_embedding_layer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from preprocess_data.tokens import OOV_TOKEN
from embeddings.embeddings import EmbeddingsType
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_image import ContinuousAttentionImageModel
from embeddings.embeddings import EmbeddingsType
from models.continuous_encoder_decoder_models.continuous_losses import ContinuousLossesType


class ContinuousAttentionImagePOSModel(ContinuousAttentionImageModel):

    def __init__(self,
                 args,
                 vocab_size,
                 token_to_id,
                 id_to_token,
                 max_len,
                 device
                 ):
        super().__init__(args, vocab_size, token_to_id, id_to_token, max_len, device)

    def _predict(self, encoder_out, caps, caption_lengths):
        caps_tokens = caps[:, 0, :].long()  # caps[:,1,:] is the pos_tagging
        return super()._predict(encoder_out, caps_tokens, caption_lengths)

    def _define_loss_criteria(self):
        loss_type = self.args.continuous_loss_type

        if loss_type == ContinuousLossesType.SMOOTHL1_AVG_SENTENCE_AND_INPUTS.value:
            self.loss_method = self.smoothl1_avg_sentence_and_inputs_loss_with_pos_tagging
            self.criterion_word_level = nn.SmoothL1Loss(reduction="none").to(self.device)
            self.criterion_sentence_level = nn.SmoothL1Loss().to(self.device)
        elif loss_type == ContinuousLossesType.SMOOTHL1_TRIPLET_AVG_SENTENCE_AND_INPUTS.value:
            self.loss_method = self.smoothl1_avg_sentence_and_inputs_loss_with_pos_tagging_all
            self.criterion_word_level = nn.SmoothL1Loss(reduction="none").to(self.device)
            self.criterion_sentence_level = nn.SmoothL1Loss().to(self.device)
        elif loss_type == ContinuousLossesType.SMOOTHL1.value:
            self.loss_method = smoothl1_pos_tagging
            self.criterion_word_level = nn.SmoothL1Loss(reduction="none").to(self.device)
        else:
            raise Exception("only available: smoothl1_avg_sentence_and_inputs_loss_with_pos_tagging ")
        # elif loss_type == ContinuousLossesType.SMOOTHL1_TRIPLET_AVG_SENTENCE_AND_INPUTS.value:
        #     self.loss_method = self.smoothl1_triplet_avg_sentence_and_inputs_loss

    def _calculate_loss(self, predict_output, caps, caption_lengths):
        predictions = predict_output["predictions"]
        caps_tokens = caps[:, 0, :].long()
        caps_pos = caps[:, 1, :]
        targets = caps_tokens[:, 1:]  # targets doesnt have stark token
        # vais buscar aqui a tua loss
        target_embeddings = self.decoder.embedding(targets).to(self.device)

        loss = self.loss_method(
            predictions,
            target_embeddings,
            caption_lengths,
            caps_pos
        )

        return loss

    def smoothl1_avg_sentence_and_inputs_loss_with_pos_tagging(
        self,
        predictions,
        target_embeddings,
        caption_lengths,
        caps_pos
    ):
        word_losses = 0.0  # pred_against_target_loss; #pred_sentence_again_target_sentence;"pred_sentence_agains_image
        sentence_losses = 0.0
        input1_losses = 0.0
        input2_losses = 0.0

        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)
        images_embedding = self.decoder.image_embedding

        n_sentences = predictions.size()[0]
        for i in range(n_sentences):  # iterate by sentence
            preds_without_padd = predictions[i, :caption_lengths[i], :]
            targets_without_padd = target_embeddings[i, :caption_lengths[i], :]
            pos_scores = caps_pos[i, :caption_lengths[i]]

            # word-level loss   (each prediction against each target)
            # for each word, have the smooth-l1 for each dim
            word_loss_per_dim = self.criterion_word_level(
                preds_without_padd,
                targets_without_padd
            )

            loss_of_each_word = torch.mean(word_loss_per_dim, dim=1)
            weighted_postagging_loss = torch.sum(loss_of_each_word*pos_scores)/torch.sum(pos_scores)
            word_losses += weighted_postagging_loss

            # sentence-level loss (sentence predicted agains target sentence)
            sentence_mean_pred = torch.mean(preds_without_padd, dim=0)  # ver a dim
            sentece_mean_target = torch.mean(targets_without_padd, dim=0)

            sentence_losses += self.criterion_sentence_level(
                sentence_mean_pred,
                sentece_mean_target
            )

            image_embedding = torch.nn.functional.normalize(images_embedding[i], p=2, dim=-1)

            # 1º input loss (sentence predicted against input image)
            input1_losses += self.criterion_sentence_level(
                sentence_mean_pred,
                image_embedding
            )

            # 2º input loss (sentence predicted against input image)
            input2_losses += self.criterion_sentence_level(
                image_embedding,
                sentece_mean_target
            )

        word_loss = word_losses/n_sentences
        sentence_loss = sentence_losses/n_sentences
        input1_loss = input1_losses/n_sentences
        input2_loss = input2_losses/n_sentences

        loss = word_loss + sentence_loss + input1_loss + input2_loss

        return loss

    def smoothl1_pos_tagging(
        self,
        predictions,
        target_embeddings,
        caption_lengths,
        caps_pos
    ):
        word_losses = 0.0  # pred_against_target_loss; #pred_sentence_again_target_sentence;"pred_sentence_agains_image

        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)

        n_sentences = predictions.size()[0]
        for i in range(n_sentences):  # iterate by sentence
            preds_without_padd = predictions[i, :caption_lengths[i], :]
            targets_without_padd = target_embeddings[i, :caption_lengths[i], :]
            pos_scores = caps_pos[i, :caption_lengths[i]]

            # word-level loss   (each prediction against each target)
            # for each word, have the smooth-l1 for each dim
            word_loss_per_dim = self.criterion_word_level(
                preds_without_padd,
                targets_without_padd
            )

            loss_of_each_word = torch.mean(word_loss_per_dim, dim=1)
            weighted_postagging_loss = torch.sum(loss_of_each_word*pos_scores)/torch.sum(pos_scores)
            word_losses += weighted_postagging_loss

        word_loss = word_losses/n_sentences
        loss = word_loss

        return loss

    def smoothl1_avg_sentence_and_inputs_loss_with_pos_tagging_all(
        self,
        predictions,
        target_embeddings,
        caption_lengths,
        caps_pos
    ):
        word_losses = 0.0  # pred_against_target_loss; #pred_sentence_again_target_sentence;"pred_sentence_agains_image
        sentence_losses = 0.0
        input1_losses = 0.0
        input2_losses = 0.0

        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)
        images_embedding = self.decoder.image_embedding

        n_sentences = predictions.size()[0]
        for i in range(n_sentences):  # iterate by sentence
            preds_without_padd = predictions[i, :caption_lengths[i], :]
            targets_without_padd = target_embeddings[i, :caption_lengths[i], :]
            pos_scores = caps_pos[i, :caption_lengths[i]].double()

            # word-level loss   (each prediction against each target)
            # Reduction "none": for each word, have the smooth-l1 for each dim
            word_loss_per_dim = self.criterion_word_level(
                preds_without_padd,
                targets_without_padd
            )

            loss_of_each_word = torch.mean(word_loss_per_dim, dim=1)
            weighted_postagging_loss = torch.sum(loss_of_each_word*pos_scores)/torch.sum(pos_scores)
            word_losses += weighted_postagging_loss

            # sentence-level loss (sentence predicted agains target sentence)
            sentence_mean_pred = torch.mean(preds_without_padd, dim=0)  # ver a dim
            # weighted_sentence target
            print("sentence mean target", torch.mean(targets_without_padd, dim=0).size())
            print(" target", targets_without_padd.size())
            print(" pos_scores", pos_scores.size())

            sentece_mean_target = torch.sum(targets_without_padd*pos_scores.unsqueeze(1), dim=0)/torch.sum(pos_scores)
            print("this is sentence mean target", sentece_mean_target.size())

            sentence_losses += self.criterion_sentence_level(
                sentence_mean_pred,
                sentece_mean_target
            )

            image_embedding = torch.nn.functional.normalize(images_embedding[i], p=2, dim=-1)

            # 1º input loss (sentence predicted against input image)
            input1_losses += self.criterion_sentence_level(
                sentence_mean_pred,
                image_embedding
            )

            # 2º input loss (sentence predicted against input image)
            input2_losses += self.criterion_sentence_level(
                image_embedding,
                sentece_mean_target
            )

        word_loss = word_losses/n_sentences
        sentence_loss = sentence_losses/n_sentences
        input1_loss = input1_losses/n_sentences
        input2_loss = input2_losses/n_sentences

        loss = word_loss + sentence_loss + input1_loss + input2_loss

        return loss
