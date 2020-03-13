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
    embedding_layer = nn.Embedding(vocab_size, embed_dim)  # embedding layer

    if embedding_type == None:
        embedding_layer.weight.data.uniform_(-0.1, 0.1)
    else:
        if embedding_type == EmbeddingsType.GLOVE.value:

            pretrained_embeddings = _get_glove_embeddings_matrix(
                vocab_size, embed_dim, token_to_id)

        elif embedding_type == EmbeddingsType.GLOVE_FOR_CONTINUOUS_MODELS.value:
            # embed_dim - 1 since it is considering end_token
            pretrained_embeddings = _get_glove_embeddings_matrix_for_continuous(
                vocab_size, embed_dim, token_to_id)

            # embedding.weight = nn.Parameter(torch.from_numpy(embeddings))

        embedding_layer.weight.data.copy_(
            torch.from_numpy(pretrained_embeddings))

    return embedding_layer


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
    embeddings_matrix = np.zeros(
        (vocab_size, embedding_size))

    #
    for word, id in token_to_id.items():
        try:
            embeddings_matrix[id] = glove_embeddings[word]
        except:
            words_unknow.append(word)
            # pass

    return embeddings_matrix

# truncate do vocabulary ->dados de treino unk
# na loss smp se a palavra de referencia é 0-> é unknow...
# durante a inferencia nnca gero unknowns...

# embeddings fixos...
# layer de embeddings
# #ignore unknown (loss=0)
# depois qd vou gerar -> matrix toda do pre-trained glove (isto...)


def _get_glove_embeddings_matrix_for_continuous(vocab_size, embedding_size, token_to_id):
    glove_path = _get_glove_path(embedding_size)

    glove_embeddings = _read_glove_vectors(
        glove_path, embedding_size)

    glove_unused_words = list(
        set(glove_embeddings.keys()) - set(token_to_id.keys()))

    glove_embeddings_unknown = []
    for word in glove_unused_words:
        glove_embeddings_unknown.append(glove_embeddings[word])
    numpy_glove_embeddings_unknown = np.asarray(
        glove_embeddings_unknown, dtype=np.float32)
    mean_glove_embeddings_unknown = np.mean(
        numpy_glove_embeddings_unknown, axis=0)

    # Init the embeddings layer with unk words having mean of embeddings unused with glove.
    embeddings_matrix = np.zeros(
        (vocab_size, embedding_size))

    for word, id in token_to_id.items():
        try:
            embeddings_matrix[id] = glove_embeddings[word]
        except:
            embeddings_matrix[id] = mean_glove_embeddings_unknown

    pont_word = "."
    embeddings_matrix[token_to_id[END_TOKEN]] = glove_embeddings[pont_word]

    return embeddings_matrix
