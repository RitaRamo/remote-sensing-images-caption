from create_data_files import get_vocab_info, get_dataset, PATH_DATASETS_RSICD
from collections import defaultdict
import spacy
from spacy.tokens import Doc
from preprocess_data.tokens import WhitespaceTokenizer
import inflect
from collections import Counter
import torch
from collections import OrderedDict
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE


VOCAB_SIZE = 512
dataset_path = "src/data/RSICD/datasets/pos_tagging_dataset"

if __name__ == "__main__":
    corpus_bigram_prob = {}

    train_dataset = get_dataset(PATH_DATASETS_RSICD+"train.json")
    vocab_info = get_vocab_info(PATH_DATASETS_RSICD+"vocab_info.json")
    vocab_size, token_to_id, id_to_token, max_len = vocab_info[
        "vocab_size"], vocab_info["token_to_id"], vocab_info["id_to_token"], vocab_info["max_len"]

    images_names, captions_of_tokens = train_dataset[
        "images_names"], train_dataset["captions_tokens"]

    print("this are caption", captions_of_tokens)

    vocab_words = token_to_id.keys()
    for word in vocab_words:
        corpus_bigram_prob[word] = {}

    n = 2
    train_data, padded_sents = padded_everygram_pipeline(n, captions_of_tokens)

    model = MLE(n)  # Lets train a 3-grams maximum likelihood estimation model.
    model.fit(train_data, padded_sents)

    for prev_word in vocab_words:
        denominator = sum(model.counts[[prev_word]].values()) + vocab_size  # add vocab for laplace smooth
        print("prev_word", prev_word)

        for word in vocab_words:
            numerator = model.counts[[prev_word]][word] + 1  # add-1 for laplace smooth
            corpus_bigram_prob[word][prev_word] = numerator/denominator

    state = {
        "corpus_bigram_prob": corpus_bigram_prob
    }

    torch.save(state, "src/data/RSICD/datasets/corpus_bigram_prob")

    # for word in vocab:
    #     corpus_prob[word]={keys:all word;=0}

    # for caption in captions_of_tokens:
    #     #Counter

    # #ficas com o counter de cada palavra

    # iterar palavra a palavra:
    # #palavra dada a anterior
    # dict[palavra][""]

    # attention
