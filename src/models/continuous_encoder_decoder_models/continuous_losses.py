from enum import Enum
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class ContinuousLossesType(Enum):
    COSINE = "cosine"
    MARGIN = "margin"


def margin_args(predictions, target_embeddings, pretrained_embedding_matrix, device):

    negative_examples = torch.zeros(target_embeddings.size()[
                                    0], target_embeddings.size()[1])

    for i in range(len(target_embeddings)):

        target_similarity_to_embeddings = cosine_similarity(target_embeddings[i].unsqueeze_(0),
                                                            pretrained_embedding_matrix)[0]

        id_of_most_similar_embedding_except_itself = np.argsort(
            target_similarity_to_embeddings)[::-1][1]

        nearest_neighbour_embedding = pretrained_embedding_matrix[
            id_of_most_similar_embedding_except_itself]

        negative_examples[i, :] = nearest_neighbour_embedding

    #print("negative examples", negative_examples)

    return predictions, target_embeddings, negative_examples.to(device)


def cosine_args(predictions, target_embeddings, pretrained_embedding_matrix, device):

    y = torch.ones(target_embeddings.shape[0]).to(device)

    return predictions, target_embeddings, y
