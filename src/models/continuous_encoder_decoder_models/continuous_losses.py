from enum import Enum
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from torch.nn import functional
from torch import nn
from utils.utils import get_pack_padded_sequences


class ContinuousLossesType(Enum):
    COSINE = "cosine"
    MARGIN = "margin"
    MARGIN_SYN_DISTANCE = "margin_syn_distance"
    MARGIN_SYN_SIMILARITY = "margin_syn_similarity"
    SMOOTHL1 = "smoothl1"
    SMOOTHL1_TRIPLET = "smoothl1_triplet"
    SMOOTHL1_TRIPLET_DIFF = "smoothl1_triplet_diff"
    SMOOTHL1_AVG_SENTENCE = "smoothl1_avg_sentence"
    #SMOOTHL1_TRIPLET_AVG_SENTENCE = "smoothl1_triplet_avg_sentence"


class ContinuousLoss():

    def __init__(self, loss_type, device):
        self.device = device

        if loss_type == ContinuousLossesType.COSINE.value:
            self.loss_method = self.cosine_loss
            self.criterion = nn.CosineEmbeddingLoss().to(self.device)

        elif loss_type == ContinuousLossesType.MARGIN.value:
            self.loss_method = self.margin_loss
            self.criterion = nn.TripletMarginLoss(
                margin=1.0, p=2).to(self.device)

        elif loss_type == ContinuousLossesType.MARGIN_SYN_DISTANCE.value:
            self.loss_method = self.margin_syn_distance_loss
            self.criterion = nn.TripletMarginLoss(
                margin=1.0, p=2).to(self.device)

        elif loss_type == ContinuousLossesType.MARGIN_SYN_SIMILARITY.value:
            self.loss_method = self.margin_syn_similarity_loss
            self.margin = 1.0

        elif loss_type == ContinuousLossesType.SMOOTHL1.value:
            self.loss_method = self.smoothl1_loss
            self.criterion = nn.SmoothL1Loss().to(self.device)

        elif loss_type == ContinuousLossesType.SMOOTHL1_TRIPLET.value:
            self.loss_method = self.smoothl1_triplet_loss
            self.criterion = nn.SmoothL1Loss(reduction='none').to(self.device)
            self.margin = 1.0

        elif loss_type == ContinuousLossesType.SMOOTHL1_TRIPLET_DIFF.value:
            self.loss_method = self.smoothl1_triplet_diff_loss
            self.criterion = nn.SmoothL1Loss(reduction='none').to(self.device)
            self.margin = 1.0

        # elif loss_type == ContinuousLossesType.SENTENCE.value:
        #     self.loss_method = self.sentence_loss
        #     self.criterion = nn.CosineEmbeddingLoss().to(self.device)

        elif loss_type == ContinuousLossesType.SMOOTHL1_AVG_SENTENCE.value:
            self.loss_method = self.smoothl1_avg_sentence_loss
            self.criterion = nn.SmoothL1Loss().to(self.device)

    def compute_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths,
        pretrained_embedding_matrix
    ):
        return self.loss_method(predictions, target_embeddings, caption_lengths, pretrained_embedding_matrix)

    def cosine_loss(
            self,
            predictions,
            target_embeddings,
            caption_lengths,
            pretrained_embedding_matrix
    ):
        predictions, target_embeddings = get_pack_padded_sequences(predictions, target_embeddings, caption_lengths)
        y = torch.ones(target_embeddings.shape[0]).to(self.device)

        return self.criterion(predictions, target_embeddings,  y)

    def margin_syn_distance_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths,
        pretrained_embedding_matrix
    ):
        predictions, target_embeddings = get_pack_padded_sequences(predictions, target_embeddings, caption_lengths)
        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)

        orthogonal_component = (predictions - torch.sum(predictions*target_embeddings,
                                                        dim=1).unsqueeze(1) * target_embeddings)

        orthogonal_negative_examples = torch.nn.functional.normalize(orthogonal_component, p=2, dim=-1)

        return self.criterion(predictions, target_embeddings, orthogonal_negative_examples.to(self.device))

    def margin_syn_similarity_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths,
        pretrained_embedding_matrix
    ):
        predictions, target_embeddings = get_pack_padded_sequences(predictions, target_embeddings, caption_lengths)
        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)

        orthogonal_component = (predictions - torch.sum(predictions*target_embeddings,
                                                        dim=1).unsqueeze(1) * target_embeddings)

        orthogonal_negative_examples = torch.nn.functional.normalize(orthogonal_component, p=2, dim=-1)

        sim_to_negative = torch.sum(predictions*orthogonal_negative_examples, dim=1)
        sim_to_target = torch.sum(predictions*target_embeddings, dim=1)

        loss = torch.clamp(self.margin + sim_to_negative - sim_to_target, min=0).mean()
        return loss

    def margin_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths,
        pretrained_embedding_matrix
    ):
        predictions, target_embeddings = get_pack_padded_sequences(predictions, target_embeddings, caption_lengths)
        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)

        negative_examples = torch.zeros(target_embeddings.size()[0], target_embeddings.size()[1])

        for i in range(len(target_embeddings)):

            diff = predictions[i] - target_embeddings[i]

            target_similarity_to_embeddings = functional.cosine_similarity(diff.unsqueeze_(0),
                                                                           pretrained_embedding_matrix)

            top_scores, top_indices = torch.topk(target_similarity_to_embeddings, k=1, dim=0)
            id_most_informative_negative = top_indices[0]
            informative_negative_embedding = pretrained_embedding_matrix[id_most_informative_negative]

            negative_examples[i, :] = informative_negative_embedding

        return self.criterion(predictions, target_embeddings,  negative_examples.to(self.device))

    def smoothl1_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths,
        pretrained_embedding_matrix
    ):
        predictions, target_embeddings = get_pack_padded_sequences(predictions, target_embeddings, caption_lengths)
        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)

        return self.criterion(predictions, target_embeddings)

    def smoothl1_triplet_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths,
        pretrained_embedding_matrix
    ):
        predictions, target_embeddings = get_pack_padded_sequences(predictions, target_embeddings, caption_lengths)
        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)

        orthogonal_component = (predictions - torch.sum(predictions*target_embeddings,
                                                        dim=1).unsqueeze(1) * target_embeddings)

        orthogonal_negative_examples = torch.nn.functional.normalize(orthogonal_component, p=2, dim=-1)

        # apply distance of smoothl1
        dist_to_negative = self.criterion(predictions, orthogonal_negative_examples)
        dist_to_target = self.criterion(predictions, target_embeddings)

        loss = torch.clamp(self.margin + dist_to_target - dist_to_negative, min=0).mean()
        return loss

    def smoothl1_triplet_diff_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths,
        pretrained_embedding_matrix
    ):
        predictions, target_embeddings = get_pack_padded_sequences(predictions, target_embeddings, caption_lengths)
        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)

        diff_component = (predictions - target_embeddings)

        diff_negative_examples = torch.nn.functional.normalize(diff_component, p=2, dim=-1)

        # apply distance of smoothl1
        dist_to_negative = self.criterion(predictions, diff_negative_examples)
        dist_to_target = self.criterion(predictions, target_embeddings)

        loss = torch.clamp(self.margin + dist_to_target - dist_to_negative, min=0).mean()
        return loss

    def sentence_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths,
        pretrained_embedding_matrix
    ):
        word_losses = 0.0

        n_sentences = predictions.size()[0]
        for i in range(n_sentences):  # iterate by sentence
            preds_without_padd = predictions[i, :caption_lengths[i], :]
            targets_without_padd = target_embeddings[i, :caption_lengths[i], :]

            y = torch.ones(targets_without_padd.shape[0])

            # word-level loss
            word_losses += self.criterion(
                preds_without_padd,
                targets_without_padd,
                y
            )

        word_loss = word_losses/n_sentences

        return word_loss

    def smoothl1_avg_sentence_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths,
        pretrained_embedding_matrix
    ):
        word_losses = 0.0
        sentence_losses = 0.0

        n_sentences = predictions.size()[0]
        for i in range(n_sentences):  # iterate by sentence
            preds_without_padd = predictions[i, :caption_lengths[i], :]
            targets_without_padd = target_embeddings[i, :caption_lengths[i], :]

            # word-level loss
            word_losses += self.criterion(
                preds_without_padd,
                targets_without_padd
            )

            # sentence-level loss
            sentence_mean_pred = torch.mean(preds_without_padd, dim=0)  # ver a dim
            sentece_mean_target = torch.mean(targets_without_padd, dim=0)

            sentence_losses += self.criterion(
                sentence_mean_pred,
                sentece_mean_target
            )

        word_loss = word_losses/n_sentences
        sentence_loss = sentence_losses/n_sentences

        loss = word_loss + sentence_loss

        return loss
