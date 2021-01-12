# até ao 12:00 acabar isto!!
import sys
sys.path.append('src/')
import json
import logging
import nltk
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from collections import defaultdict
from data_preprocessing.preprocess_tokens import END_TOKEN, START_TOKEN, preprocess_tokens
import re
from definitions_datasets import get_dataset_paths
from datetime import datetime


# def _get_images_and_captions(dataset):
#     images_names = {"train": [], "val": [], "test": []}
#     captions_of_tokens = {"train": [], "val": [], "test": []}
#     for row in dataset["images"]:
#         image_name = row["filename"]
#         split = row["split"]

#         for caption in row["sentences"]:
#             if not caption["tokens"]:
#                 continue
#             if caption["tokens"][-1] == ".":
#                 if len(caption["tokens"][:-1]) > 0:  # len sentence without pont needs to >0 to be considered
#                     caption_tokens = caption["tokens"][:-1]
#                 else:
#                     continue
#             else:
#                 caption_tokens = caption["tokens"]

#             tokens = [START_TOKEN] + [token.lower() for token in caption_tokens] + [END_TOKEN]

#             print("this is the token for image fianl", tokens, image_name)

#             captions_of_tokens[split].append(tokens)
#             images_names[split].append(image_name)

#     return images_names, captions_of_tokens


def _get_images_and_captions(dataset):
    images_names = {"train": [], "val": [], "test": []}
    captions_of_tokens = {"train": [], "val": [], "test": []}
    all_captions_of_tokens = {"train": [], "val": [], "test": []}
    a = 0
    for row in dataset["images"]:
        image_name = row["filename"]
        split = row["split"]

        all_caps = []
        n_captions = 0
        for caption in row["sentences"]:
            if not caption["tokens"]:
                continue
            if caption["tokens"][-1] == ".":
                if len(caption["tokens"][:-1]) > 0:  # len sentence without pont needs to >0 to be considered
                    caption_tokens = caption["tokens"][:-1]
                else:
                    continue
            else:
                caption_tokens = caption["tokens"]

            tokens = [START_TOKEN] + [token.lower() for token in caption_tokens] + [END_TOKEN]
            all_caps.append(tokens)
            #print("this is the token for image fianl", tokens, image_name)

            captions_of_tokens[split].append(tokens)
            images_names[split].append(image_name)
            n_captions += 1

        for _ in range(n_captions):  # for each image, and caption, have also 5 captions
            all_captions_of_tokens[split].append(all_caps)

    print("images_names len", len(images_names["train"]))
    print("captions len", len(captions_of_tokens["train"]))
    print("captions len", len(all_captions_of_tokens["train"]))

    print("images_names len v", len(images_names["val"]))
    print("captions len", len(captions_of_tokens["val"]))
    print("captions len", len(all_captions_of_tokens["val"]))
    return images_names, captions_of_tokens, all_captions_of_tokens


def _get_dict_image_and_its_captions(dataset, split="test"):
    images_captions = defaultdict(list)
    for row in dataset["images"]:
        image_name = row["filename"]
        image_id = row["imgid"]
        if row["split"] == split:
            for caption in row["sentences"]:
                if not caption["tokens"]:
                    continue
                if caption["tokens"][-1] == ".":
                    if len(caption["tokens"][:-1]) > 0:  # len sentence without pont needs to >0 to be considered
                        caption_tokens = caption["tokens"][:-1]
                    else:
                        continue
                        #caption_tokens = caption["tokens"][:-1]
                else:
                    caption_tokens = caption["tokens"]
                tokens = [token.lower() for token in caption_tokens]
                tokens = " ".join(tokens)

                images_captions[image_name].append(tokens)

    return images_captions


def _get_test_with_coco_format(raw_dataset, split="test", split2=None):
    images = []
    annotations = []
    id_annotation = 0

    images_captions = defaultdict(list)
    for row in raw_dataset["images"]:
        image_name = row["filename"]
        image_id = row["imgid"]
        if row["split"] == split or row["split"] == split2:

            images.append(
                {
                    "id": image_id,
                    "width": 0,
                    "height": 0,
                    "file_name": image_name,
                    "license": 1,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": str(datetime.now()),
                }
            )

            for caption in row["sentences"]:
                if not caption["tokens"]:
                    continue
                if caption["tokens"][-1] == ".":
                    if len(caption["tokens"][:-1]) > 0:  # len sentence without pont needs to >0 to be considered
                        caption_tokens = caption["tokens"][:-1]
                    else:
                        continue
                        # caption_tokens = caption["tokens"][:-1]
                else:
                    caption_tokens = caption["tokens"]
                tokens = [token.lower() for token in caption_tokens]
                tokens = " ".join(tokens)

                annotations.append(
                    {
                        "id": id_annotation,
                        "image_id": image_id,
                        "caption": tokens,
                    }
                )

                id_annotation += 1

            # if id_annotation >= 11:
            #     break

    info = {
        "year": 2020,
        "version": 1,
        "description": "none",
        "contributor": "none",
        "url": "none",
        "date_created": str(datetime.now()),
    }

    licenses = [
        {
            "id": 1,
            "name": "",
            "url": ""
        }
    ]

    test_coco_format = {
        "info": info,
        "images": images,
        "annotations": annotations,
        "licenses": licenses,
        "type": "captions"
    }

    return test_coco_format


def _dump_dict_to_json(dict, file_dir, file_name):
    with open(file_dir + file_name, 'w+') as f:
        json.dump(dict, f, indent=2)


def _dump_data_to_json(images_names, captions_tokens, file_dir, file_name, all_captions=None):
    if all_captions:
        dataset_dict = {
            "images_names": images_names,
            "captions_tokens": captions_tokens,
            "all_captions_tokens": all_captions
        }
    else:
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
    print("hay")
    images_names, captions_of_tokens, all_captions_of_tokens = _get_images_and_captions(raw_dataset)
    print("sai")
    train_images_names, train_captions_of_tokens = shuffle(
        images_names["train"], captions_of_tokens["train"], random_state=42)
    # val_images_names, val_captions_of_tokens = shuffle(
    #     images_names["val"], captions_of_tokens["val"], random_state=42)
    val_images_names, val_captions_of_tokens, val_all_captions_of_tokens = shuffle(
        images_names["val"], captions_of_tokens["val"], all_captions_of_tokens["val"], random_state=42)

    test_dict_image_captions = _get_dict_image_and_its_captions(raw_dataset)
    # only for neighbour model
    train_dict_image_captions = _get_dict_image_and_its_captions(raw_dataset, "train")

    test_coco_format = _get_test_with_coco_format(raw_dataset, split="test")
    val_coco_format = _get_test_with_coco_format(raw_dataset, split="val")
    train_coco_format = _get_test_with_coco_format(raw_dataset, split="train")
    #train_and_val_coco_format = _get_test_with_coco_format(raw_dataset, split="train", split2="val")

    vocab_size, token_to_id, id_to_token, max_len = preprocess_tokens(
        train_captions_of_tokens
    )  # preprocess should be done with trainset

    # save vocab and datasets
    _dump_vocab_to_json(vocab_size, token_to_id,
                        id_to_token, max_len, file_dir)

    _dump_data_to_json(train_images_names, train_captions_of_tokens,
                       file_dir, "train.json")
    _dump_data_to_json(val_images_names, val_captions_of_tokens,
                       file_dir, "val.json", val_all_captions_of_tokens)

    _dump_dict_to_json(test_dict_image_captions, file_dir, "test.json")
    _dump_dict_to_json(train_dict_image_captions, file_dir, "train_dict.json")

    _dump_dict_to_json(test_coco_format, file_dir, "test_coco_format.json")
    _dump_dict_to_json(val_coco_format, file_dir, "val_coco_format.json")
    _dump_dict_to_json(train_coco_format, file_dir, "train_coco_format.json")
    #_dump_dict_to_json(train_and_val_coco_format, file_dir, "train_and_val_coco_format.json")


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

    logging.info("start to save datasets and vocab of RSCID")
    nltk.download('wordnet')
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    # raw_dataset = pd.read_json(PATH_RSICD + "raw_dataset/dataset_rsicd.json")
    # _save_dataset(raw_dataset, PATH_DATASETS_RSICD)

    # raw_dataset = pd.read_json(PATH_UCM + "raw_dataset/dataset.json")
    # _save_dataset(raw_dataset, PATH_DATASETS_UCM)

    logging.info("saving datasets and vocab of RSICD")

    dataset_folder, dataset_jsons = get_dataset_paths("rsicd")
    raw_dataset = pd.read_json(dataset_folder + "raw_dataset/dataset.json")
    _save_dataset(raw_dataset, dataset_jsons)

    logging.info("saving datasets and vocab of UCM")

    dataset_folder, dataset_jsons = get_dataset_paths("ucm")
    raw_dataset = pd.read_json(dataset_folder + "raw_dataset/dataset.json")
    _save_dataset(raw_dataset, dataset_jsons)

    logging.info("saving datasets and vocab of Flickr8k")

    dataset_folder, dataset_jsons = get_dataset_paths("flickr8k")
    raw_dataset = pd.read_json(dataset_folder + "raw_dataset/dataset.json")
    _save_dataset(raw_dataset, dataset_jsons)

    logging.info("saved datasets and vocab")

    logging.info("saving datasets and vocab of COCO")

    dataset_folder, dataset_jsons = get_dataset_paths("COCO")
    raw_dataset = pd.read_json(dataset_folder + "raw_dataset/dataset.json")
    _save_dataset(raw_dataset, dataset_jsons)

    logging.info("saved datasets and vocab")


