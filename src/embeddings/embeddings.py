import numpy as np
from enum import Enum
import spacy
from preprocess_data.tokens import END_TOKEN
import torch
from torch import nn
import io
import logging


class EmbeddingsType(Enum):
    GLOVE = "glove"
    FASTTEXT = "fasttext"


def get_embedding_layer(embedding_type, embed_dim, vocab_size, token_to_id):
    embedding_layer = nn.Embedding(vocab_size, embed_dim)  # embedding layer

    if embedding_type == None:
        logging.info("inicialize embeddings")
        embedding_layer.weight.data.uniform_(-0.1, 0.1)
    else:
        if embedding_type == EmbeddingsType.GLOVE.value:
            logging.info("loading pretrained embeddings of glove")

            pretrained_embeddings = _get_glove_embeddings_matrix(
                vocab_size, embed_dim, token_to_id)

        elif embedding_type == EmbeddingsType.FASTTEXT.value:
            logging.info("loading pretrained embeddings of fasttext")

            pretrained_embeddings = _get_fasttext_embeddings_matrix(
                vocab_size, embed_dim, token_to_id)

        embedding_layer.weight.data.copy_(
            torch.from_numpy(pretrained_embeddings))

        # pretrained embedings are not trainable by default
        embedding_layer.weight.requires_grad = False

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


def _read_fasttext_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


def _get_glove_path(embedding_size):
    return 'src/embeddings/glove.6B/glove.6B.'+str(embedding_size) + 'd.txt'


def _get_fasttext_path(embedding_size):
    return 'src/embeddings/faxttext/wiki-news-300d-1M-subword.vec'

# truncate do vocabulary ->dados de treino unk
# na loss smp se a palavra de referencia é 0-> é unknow...
# durante a inferencia nnca gero unknowns...

# embeddings fixos...
# layer de embeddings
# #ignore unknown (loss=0)
# depois qd vou gerar -> matrix toda do pre-trained glove (isto...)


def _get_glove_embeddings_matrix(vocab_size, embedding_size, token_to_id):
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

    embeddings_matrix[token_to_id[END_TOKEN]] = glove_embeddings["."]

    return embeddings_matrix


def _get_fasttext_embeddings_matrix(vocab_size, embedding_size, token_to_id):
    glove_path = _get_fasttext_path(embedding_size)

    glove_embeddings = _read_fasttext_vectors(
        glove_path)

    print("this is fasttext", glove_embeddings)

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
