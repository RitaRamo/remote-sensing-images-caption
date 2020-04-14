import numpy as np
from enum import Enum
import spacy
from preprocess_data.tokens import END_TOKEN
import torch
from torch import nn
import io
import logging
import fasttext


class EmbeddingsType(Enum):
    GLOVE = "glove"
    FASTTEXT = "fasttext"
    CONCATENATE_GLOVE_FASTTEXT = "concatenate_glove_fasttext"


def get_embedding_layer(embedding_type, embed_dim, vocab_size, token_to_id):
    embedding_layer = nn.Embedding(vocab_size, embed_dim)  # embedding layer

    if embedding_type == None:
        logging.info("inicialize embeddings")
        embedding_layer.weight.data.uniform_(-0.1, 0.1)
    else:
        if embedding_type == EmbeddingsType.GLOVE.value:
            logging.info("loading pretrained embeddings of glove")

            # pretrained_embeddings = _get_glove_embeddings_matrix(
            #     vocab_size, embed_dim, token_to_id)

            glove_path = _get_glove_path(embed_dim)

            glove_embeddings = _read_glove_vectors(
                glove_path, embed_dim)

            pretrained_embeddings = _get_embeddings_matrix(
                glove_embeddings, vocab_size, embed_dim, token_to_id)

        elif embedding_type == EmbeddingsType.FASTTEXT.value:
            logging.info("loading pretrained embeddings of fasttext")

            # pretrained_embeddings = _get_fasttext_embeddings_matrix(
            #     vocab_size, embed_dim, token_to_id)

            # fasttext_path = _get_fasttext_path(embed_dim)

            # fasttext_embeddings = _read_fasttext_vectors(
            #     fasttext_path)

            fasttext_path = _get_fasttext_path(embed_dim)

            fasttext_embeddings = fasttext.load_model(fasttext_path)

            pretrained_embeddings = _get_fasttext_embeddings_matrix(
                fasttext_embeddings, vocab_size, embed_dim, token_to_id)

        elif embedding_type == EmbeddingsType.CONCATENATE_GLOVE_FASTTEXT.value:

            glove_path = _get_glove_path(embed_dim)

            glove_embeddings = _read_glove_vectors(
                glove_path, embed_dim)

            glove_pretrained_embeddings = _get_embeddings_matrix(
                glove_embeddings, vocab_size, embed_dim, token_to_id)

            fasttext_path = _get_fasttext_path(embed_dim)

            fasttext_embeddings = fasttext.load_model(fasttext_path)

            fasttext_pretrained_embeddings = _get_fasttext_embeddings_matrix(
                fasttext_embeddings, vocab_size, embed_dim, token_to_id)

            np.concatenate((glove_pretrained_embeddings,
                            fasttext_pretrained_embeddings), axis=1)

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
        data[tokens[0]] = tokens[1:]

    return data


def _get_glove_path(embedding_size):
    return 'src/embeddings/glove.6B/glove.6B.'+str(embedding_size) + 'd.txt'


def _get_fasttext_path(embedding_size):
    # source: https://github.com/facebookresearch/fastText/blob/master/docs/pretrained-vectors.md
    return 'src/embeddings/fasttext/wiki.en.bin'


def _get_embeddings_matrix(embeddings, vocab_size, embedding_size, token_to_id):
    # reduce the matrix of pretrained:embeddings according to dataset vocab

    unused_words = list(
        set(embeddings.keys()) - set(token_to_id.keys()))

    embeddings_unknown = []

    for word in unused_words:
        embeddings_unknown.append(embeddings[word])
    numpy_embeddings_unknown = np.asarray(
        embeddings_unknown, dtype=np.float32)

    mean_embeddings_unknown = np.mean(
        numpy_embeddings_unknown, axis=0)

    # Init the embeddings layer with unk words having mean of embeddings unused with glove.
    embeddings_matrix = np.zeros(
        (vocab_size, embedding_size))

    for word, id in token_to_id.items():
        try:
            embeddings_matrix[id] = embeddings[word]
        except:
            embeddings_matrix[id] = mean_embeddings_unknown

    embeddings_matrix[token_to_id[END_TOKEN]] = embeddings["."]

    return embeddings_matrix


def _get_fasttext_embeddings_matrix(embeddings, vocab_size, embedding_size, token_to_id):
    # reduce the matrix of pretrained:embeddings according to dataset vocab
    print("eheh entrei aqui")

    embeddings_matrix = np.zeros(
        (vocab_size, embedding_size))

    for word, id in token_to_id.items():
        try:
            embeddings_matrix[id] = embeddings.get_word_vector(word)
        except:
            print("entrei aqui como?")
            pass

    return embeddings_matrix


# def _get_glove_embeddings_matrix(vocab_size, embedding_size, token_to_id):
#     glove_path = _get_glove_path(embedding_size)

#     glove_embeddings = _read_glove_vectors(
#         glove_path, embedding_size)

#     glove_unused_words = list(
#         set(glove_embeddings.keys()) - set(token_to_id.keys()))

#     glove_embeddings_unknown = []

#     for word in glove_unused_words:
#         glove_embeddings_unknown.append(glove_embeddings[word])
#     numpy_glove_embeddings_unknown = np.asarray(
#         glove_embeddings_unknown, dtype=np.float32)

#     mean_glove_embeddings_unknown = np.mean(
#         numpy_glove_embeddings_unknown, axis=0)

#     # Init the embeddings layer with unk words having mean of embeddings unused with glove.
#     embeddings_matrix = np.zeros(
#         (vocab_size, embedding_size))

#     for word, id in token_to_id.items():
#         try:
#             embeddings_matrix[id] = glove_embeddings[word]
#         except:
#             embeddings_matrix[id] = mean_glove_embeddings_unknown

#     embeddings_matrix[token_to_id[END_TOKEN]] = glove_embeddings["."]

#     return embeddings_matrix


# def _get_fasttext_embeddings_matrix(vocab_size, embedding_size, token_to_id):
#     glove_path = _get_fasttext_path(embedding_size)

#     glove_embeddings = _read_fasttext_vectors(
#         glove_path)

#     print("this is fasttext", glove_embeddings)

#     glove_unused_words = list(
#         set(glove_embeddings.keys()) - set(token_to_id.keys()))

#     glove_embeddings_unknown = []
#     for word in glove_unused_words:
#         glove_embeddings_unknown.append(glove_embeddings[word])
#     numpy_glove_embeddings_unknown = np.asarray(
#         glove_embeddings_unknown, dtype=np.float32)
#     mean_glove_embeddings_unknown = np.mean(
#         numpy_glove_embeddings_unknown, axis=0)

#     # Init the embeddings layer with unk words having mean of embeddings unused with glove.
#     embeddings_matrix = np.zeros(
#         (vocab_size, embedding_size))

#     for word, id in token_to_id.items():
#         try:
#             embeddings_matrix[id] = glove_embeddings[word]
#         except:
#             embeddings_matrix[id] = mean_glove_embeddings_unknown

#     # pont_word = "."
#     # embeddings_matrix[token_to_id[END_TOKEN]] = glove_embeddings[pont_word]

#     print("this is embedding matrix", embeddings_matrix)

#     return embeddings_matrix
