import numpy as np
from enum import Enum
import spacy
from preprocess_data.tokens import END_TOKEN
import torch
from torch import nn


class EmbeddingsType(Enum):
    GLOVE = "glove"
    GLOVE_FOR_CONTINUOUS_MODELS = "glove_continuous"


def get_embedding_layer(embedding_type, embed_dim, vocab_size, token_to_id):
    embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer

    if embedding_type == None:
        embedding.weight.data.uniform_(-0.1, 0.1)
    else:
        if embedding_type == EmbeddingsType.GLOVE.value:

            embeddings = _get_glove_embeddings_matrix(
                vocab_size, embed_dim, token_to_id)

        elif embedding_type == EmbeddingsType.GLOVE_FOR_CONTINUOUS_MODELS.value:
            # embed_dim - 1 since it is considering end_token
            embeddings = _get_glove_embeddings_matrix_for_continuous(
                vocab_size, embed_dim-1, token_to_id)

        embedding.weight = nn.Parameter(embeddings)

    return embedding


def _read_glove_vectors(path, lenght):
    embeddings = {}
    with open(path) as glove_f:
        for line in glove_f:
            chunks = line.split()
            word = chunks[0]
            vector = np.array(chunks[1:])
            embeddings[word] = vector

    return embeddings


def _get_glove_path(embedding_size):
    return 'src/embeddings/glove.6B/glove.6B.'+str(embedding_size) + 'd.txt'


def _get_glove_embeddings_matrix(vocab_size, embedding_size, token_to_id):
    # ter caderno, fazer a logica, ver outros problemas...

    glove_path = _get_glove_path(embedding_size)

    glove_embeddings = _read_glove_vectors(
        glove_path, embedding_size)

    words_unknow = []
    # Init the embeddings layer
    embeddings_matrix = torch.zeros(
        (vocab_size, embedding_size))
    for word, id in token_to_id.items():
        try:
            embeddings_matrix[id] = glove_embeddings[word]
        except:
            words_unknow.append(word)
            # pass

    return embeddings_matrix


def _get_glove_embeddings_matrix_for_continuous(vocab_size, embedding_size, token_to_id):
    glove_path = _get_glove_path(embedding_size)

    glove_embeddings = _read_glove_vectors(
        glove_path, embedding_size)

    embedding_size = embedding_size+1  # add one dim for the END_TOKEN

    # Init the embeddings layer
    embeddings_matrix = torch.zeros(
        (vocab_size, embedding_size))

    print("np size of emebedding", np.shape(glove_embeddings))
    # print("this is np mean", np.mean(glove_embeddings,dim=1)

    embeddings_matrix[token_to_id[END_TOKEN], -1] = 1

    for word, id in token_to_id.items():
        try:
            embeddings_matrix[id, :-1] = glove_embeddings[word]
        except:
            pass

    return embeddings_matrix
