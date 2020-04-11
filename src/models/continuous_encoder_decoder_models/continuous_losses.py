from enum import Enum
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from torch.nn import functional
from torch import nn


class ContinuousLossesType(Enum):
    COSINE = "cosine"
    MARGIN = "margin"
    MARGIN_SYN_DISTANCE = "margin_syn_distance"
    MARGIN_SYN_SIMILARITY = "margin_syn_similarity"
    SMOOTHL1 = "smoothl1"


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

    def compute_loss(self, predictions, target_embeddings, pretrained_embedding_matrix):
        return self.loss_method(predictions, target_embeddings, pretrained_embedding_matrix)

    def cosine_loss(self, predictions, target_embeddings, pretrained_embedding_matrix):

        y = torch.ones(target_embeddings.shape[0]).to(self.device)

        return self.criterion(predictions, target_embeddings, y)

    def margin_syn_distance_loss(self, predictions, target_embeddings, pretrained_embedding_matrix):

        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)

        orthogonal_negative_examples = predictions - torch.sum(predictions*target_embeddings,
                                                               dim=1).unsqueeze(1) * target_embeddings

        return self.criterion(predictions, target_embeddings, orthogonal_negative_examples.to(self.device))

    def margin_syn_similarity_loss(self, predictions, target_embeddings, pretrained_embedding_matrix):

        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)

        orthogonal_negative_examples = (predictions - torch.sum(predictions*target_embeddings,
                                                                dim=1).unsqueeze(1) * target_embeddings).to(self.device)

        sim_to_negative = torch.sum(
            predictions*orthogonal_negative_examples, dim=1)
        sim_to_target = torch.sum(predictions*target_embeddings, dim=1)

        loss = torch.clamp(self.margin + sim_to_negative -
                           sim_to_target, min=0).mean()

        return loss

    def margin_loss(self, predictions, target_embeddings, pretrained_embedding_matrix):
        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)

        negative_examples = torch.zeros(target_embeddings.size()[
                                        0], target_embeddings.size()[1])

        for i in range(len(target_embeddings)):

            diff = predictions[i] - target_embeddings[i]

            target_similarity_to_embeddings = functional.cosine_similarity(diff.unsqueeze_(0),
                                                                           pretrained_embedding_matrix)

            top_scores, top_indices = torch.topk(
                target_similarity_to_embeddings, k=1, dim=0)

            id_most_informative_negative = top_indices[0]

            informative_negative_embedding = pretrained_embedding_matrix[
                id_most_informative_negative]

            negative_examples[i, :] = informative_negative_embedding

        return self.criterion(predictions, target_embeddings, negative_examples.to(self.device))

    def smoothl1_loss(self, predictions, target_embeddings, pretrained_embedding_matrix):
        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)

        return self.criterion(predictions, target_embeddings)
