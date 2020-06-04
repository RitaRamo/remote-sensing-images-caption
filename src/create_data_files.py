# até ao 12:00 acabar isto!!
import json
import logging
import nltk
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from collections import defaultdict
from preprocess_data.tokens import END_TOKEN, START_TOKEN, preprocess_tokens
import re


PATH_RSICD = "src/data/RSICD/"
PATH_DATASETS_RSICD = PATH_RSICD+"datasets/"


def _get_images_and_captions(dataset):
    images_names = {"train": [], "val": [], "test": []}
    captions_of_tokens = {"train": [], "val": [], "test": []}
    for row in dataset["images"]:
        image_name = row["filename"]
        split = row["split"]

        for caption in row["sentences"]:
            tokens = [START_TOKEN] + caption["tokens"] + [END_TOKEN]

            captions_of_tokens[split].append(tokens)
            images_names[split].append(image_name)

    return images_names, captions_of_tokens


def _get_dict_image_and_its_captions(dataset):
    images_captions = defaultdict(list)
    for row in dataset["images"]:
        image_name = row["filename"]
        if row["split"] == "test":
            for caption in row["sentences"]:
                tokens = [START_TOKEN] + caption["tokens"] + [END_TOKEN]
                tokens = " ".join(tokens)

                images_captions[image_name].append(tokens)

    return images_captions


def _dump_dict_to_json(dict, file_dir, file_name):
    with open(file_dir+file_name, 'w+') as f:
        json.dump(dict, f, indent=2)


def _dump_data_to_json(images_names, captions_tokens, file_dir, file_name):
    dataset_dict = {
        "images_names": images_names,
        "captions_tokens": captions_tokens
    }
    # falta directori
    _dump_dict_to_json(dataset_dict, file_dir, file_name)
    # with open(file_dir+file_name, 'w+') as f:
    #     json.dump(dataset_dict, f, indent=2)


def _dump_vocab_to_json(vocab_size, token_to_id, id_to_token, max_len, file_dir):
    vocab_info = {}
    vocab_info["vocab_size"] = vocab_size
    vocab_info["token_to_id"] = token_to_id
    vocab_info["id_to_token"] = id_to_token
    vocab_info["max_len"] = max_len

    _dump_dict_to_json(vocab_info, file_dir, "vocab_info.json")

    # with open(file_dir+"vocab_info.json", 'w+') as f:
    #     json.dump(vocab_info, f, indent=2)


def _save_dataset(raw_dataset, file_dir):
    # suffle and split dataset into train,val and test
    images_names, captions_of_tokens = _get_images_and_captions(raw_dataset)

    train_images_names, train_captions_of_tokens = shuffle(
        images_names["train"], captions_of_tokens["train"], random_state=42)
    val_images_names, val_captions_of_tokens = shuffle(images_names["val"], captions_of_tokens["val"], random_state=42)

    test_dict_image_captions = _get_dict_image_and_its_captions(raw_dataset)

    vocab_size, token_to_id, id_to_token, max_len = preprocess_tokens(
        train_captions_of_tokens
    )  # preprocess should be done with trainset

    vocab_size, token_to_id, id_to_token, max_len = preprocess_tokens(
        train_captions_of_tokens
    )  # preprocess should be done with trainset

    # save vocab and datasets
    _dump_vocab_to_json(vocab_size, token_to_id,
                        id_to_token, max_len, file_dir)

    _dump_data_to_json(train_images_names, train_captions_of_tokens,
                       file_dir, "train.json")
    _dump_data_to_json(val_images_names, val_captions_of_tokens,
                       file_dir, "val.json")

    _dump_dict_to_json(test_dict_image_captions, file_dir, "test.json")


def get_dataset(file_path):
    with open(file_path) as json_file:
        dataset = json.load(json_file)
    return dataset


def get_vocab_info(file_path):
    with open(file_path) as json_file:
        vocab_info = json.load(json_file)

    # given it was loaded from a json, the dict id_to_token has keys as strings instead of int, as supposed. To fix:
    vocab_info["id_to_token"] = {
        int(k): v for k, v in vocab_info["id_to_token"].items()}

    return vocab_info


if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)

    logging.info("start to save datasets and vocab of of RSCID")
    nltk.download('wordnet')
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    raw_dataset = pd.read_json(PATH_RSICD + "raw_dataset/dataset_rsicd.json")
    _save_dataset(raw_dataset, PATH_DATASETS_RSICD)

    logging.info("saved datasets and vocab")
