import os
import numpy as np
from collections import OrderedDict
from toolz.itertoolz import unique
from enum import Enum
import spacy
from spacy.tokens import Doc

START_TOKEN = "<start_seq>"
END_TOKEN = "<end_seq>"
PAD_TOKEN = "#"
OOV_TOKEN = "<unk>"
os.environ['PYTHONHASHSEED'] = '0'


def preprocess_tokens(train_captions):
    all_tokens = [START_TOKEN, END_TOKEN, OOV_TOKEN]
    for caption_tokens in train_captions:
        all_tokens.extend(caption_tokens)

    # vocab = list(set(all_tokens))
    vocab = list(unique(all_tokens))
    token_to_id = OrderedDict([(value, index+1)
                               for index, value in enumerate(vocab)])
    id_to_token = OrderedDict([(index+1, value)
                               for index, value in enumerate(vocab)])

    token_to_id[PAD_TOKEN] = 0
    id_to_token[0] = PAD_TOKEN

    len_vocab = len(vocab) + 1  # padding token

    max_len = max(map(len, train_captions))

    return len_vocab, token_to_id, id_to_token, max_len


def convert_captions_to_Y(captions_of_tokens, max_len, token_to_id):
    len_captions = []

    input_captions = np.zeros(
        (len(captions_of_tokens), max_len)) + token_to_id[PAD_TOKEN]

    for i in range(len(captions_of_tokens)):

        tokens_to_integer = [token_to_id.get(
            token, token_to_id[OOV_TOKEN]) for token in captions_of_tokens[i]]

        caption = tokens_to_integer[:max_len]

        input_captions[i, :len(caption)] = caption

        len_captions.append(len(caption))

    return input_captions, len_captions


def get_pos_score(pos):
    if pos == "NOUN":
        return 1.0
    elif pos == "ADJ":
        return 0.5
    elif pos == "VERB":
        return 0.5
    else:
        return 0.25


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


def convert_captions_to_Y_and_POS(captions_of_tokens, max_len, token_to_id):
    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

    # middle_dim=2 since we will have tokens and pos_tag
    input_captions = np.zeros(
        (len(captions_of_tokens), 2, max_len)) + token_to_id[PAD_TOKEN]
    len_captions = []

    for i in range(len(captions_of_tokens)):

        tokens_to_integer = [token_to_id.get(
            token, token_to_id[OOV_TOKEN]) for token in captions_of_tokens[i]]

        pos_to_integer = []
        # doesnot consider start token and end for spacy
        tokens_without_special_tokens = captions_of_tokens[i][1:-1]
        sentence = " ".join(tokens_without_special_tokens)
        doc = nlp(sentence) if sentence != "" else []

        # pos tagging of start_token
        pos_to_integer.append(get_pos_score(None))

        for spacy_token in doc:
            pos = spacy_token.pos_
            pos_score = get_pos_score(pos)
            pos_to_integer.append(pos_score)

        # pos tagging of end_token
        pos_to_integer.append(get_pos_score(None))

        if len(tokens_to_integer) != len(pos_to_integer):
            raise Exception("Tokens and respective pos tagging should have same len")

        caption = tokens_to_integer[:max_len]
        pos = pos_to_integer[:max_len]
        input_captions[i, 0, :len(caption)] = caption
        input_captions[i, 1, :len(pos)] = pos

        len_captions.append(len(caption))

    return input_captions, len_captions
