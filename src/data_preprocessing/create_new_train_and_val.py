# até ao 12:00 acabar isto!!
import sys
sys.path.append('src/')
import json
import logging
from definitions import PATH_RSICD, PATH_DATASETS_RSICD
from data_preprocessing.preprocess_tokens import END_TOKEN, START_TOKEN, preprocess_tokens
from data_preprocessing.create_data_files import get_dataset, _dump_data_to_json, _dump_vocab_to_json
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import defaultdict
from datetime import datetime


def _get_images_and_captions(dataset_train, dataset_val):
    images_names = {"train": [], "val": []}
    captions_of_tokens = {"train": [], "val": []}

    for split, dataset in [("train", dataset_train), ("val", dataset_val)]:

        for row in dataset:
            image_name = row[2]

            for caption in row[3]:

                caption_tokens = caption.split()
                tokens = [START_TOKEN] + caption_tokens + [END_TOKEN]

                captions_of_tokens[split].append(tokens)
                images_names[split].append(image_name)

    return images_names, captions_of_tokens


def _transform_to_coco_format(dataset):
    images = []
    annotations = []
    id_annotation = 0

    images_captions = defaultdict(list)
    for row in dataset:
        image_name = row[2]
        image_id = row[1]

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

        for caption in row[3]:
            annotations.append(
                {
                    "id": id_annotation,
                    "image_id": image_id,
                    "caption": caption,
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

    coco_format = {
        "info": info,
        "images": images,
        "annotations": annotations,
        "licenses": licenses,
        "type": "captions"
    }

    return coco_format


if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)

    logging.info("start to save datasets and vocab of of RSCID")

    # get original test dataset and save in the new folder: datasets_new_train_and_val
    test_dataset = get_dataset(PATH_DATASETS_RSICD + "test.json")
    test_coco_format = get_dataset(PATH_DATASETS_RSICD + "test_coco_format.json")

    print("coco formal", test_coco_format)

    with open(PATH_RSICD + "datasets_new_train_and_val/" + "test.json", 'w+') as f:
        json.dump(test_coco_format, f, indent=2)
    with open(PATH_RSICD + "datasets_new_train_and_val/" + "test_coco_format.json", 'w+') as f:
        json.dump(test_coco_format, f, indent=2)

    # pick the original train dataset, to change the train and val dataset
    dataset_original_train = get_dataset(PATH_DATASETS_RSICD + "train_coco_format.json")
    images_id_and_filename = [(annot["file_name"], annot["id"]) for annot in dataset_original_train["images"]]

    logging.info("splitting training dataset into 10 % to get new val split")
    classes_train = defaultdict(list)
    for i in range(len(images_id_and_filename)):
        row = images_id_and_filename[i]
        name = row[0]
        img_id = row[1]

        captions_of_img_id = []
        for ann in dataset_original_train["annotations"]:
            if ann["image_id"] == img_id:
                captions_of_img_id.append(ann["caption"])
        # append image class (obtained by the name ex: farmleand_111.jpeg)
        name_splited = name.split("_")
        if len(name_splited) > 1:
            img_class = name_splited[0]
            # image_categories[name].append(img_class)
            classes_train[img_class].append((img_class, img_id, name, captions_of_img_id))
        else:
            classes_train["unk_class"].append(("unk_class", img_id, name, captions_of_img_id))

    # convert the original training dataset into two splits: new_train and new_val, spliting by 10% for each class
    new_train_dataset = []
    new_val_dataset = []
    for class_type in list(classes_train.keys()):
        class_train, class_val = train_test_split(classes_train[class_type], test_size=0.10, random_state=42)
        new_train_dataset.extend(class_train)
        new_val_dataset.extend(class_val)

    logging.info("preprocessing train and val (start and end token, suffle data) and vocab")

    images_names, captions_of_tokens = _get_images_and_captions(new_train_dataset, new_val_dataset)

    train_images_names, train_captions_of_tokens = shuffle(
        images_names["train"], captions_of_tokens["train"], random_state=42)
    val_images_names, val_captions_of_tokens = shuffle(images_names["val"], captions_of_tokens["val"], random_state=42)

    vocab_size, token_to_id, id_to_token, max_len = preprocess_tokens(
        train_captions_of_tokens
    )  # preprocess should be done with trainset

    # save vocab and datasets
    _dump_vocab_to_json(vocab_size, token_to_id,
                        id_to_token, max_len, PATH_RSICD + "datasets_new_train_and_val/")

    _dump_data_to_json(train_images_names, train_captions_of_tokens,
                       PATH_RSICD + "datasets_new_train_and_val/", "train.json")
    _dump_data_to_json(val_images_names, val_captions_of_tokens,
                       PATH_RSICD + "datasets_new_train_and_val/", "val.json")

    logging.info("converting new train and val split into coco format")
    new_train_coco_format = _transform_to_coco_format(new_train_dataset)
    new_val_coco_format = _transform_to_coco_format(new_val_dataset)

    with open(PATH_RSICD + "datasets_new_train_and_val/" + "train_coco_format.json", 'w+') as f:
        json.dump(new_train_coco_format, f, indent=2)

    with open(PATH_RSICD + "datasets_new_train_and_val/" + "val_coco_format.json", 'w+') as f:
        json.dump(new_val_coco_format, f, indent=2)

    # vocab_size, token_to_id, id_to_token, max_len = preprocess_tokens(
    #     train_captions_of_tokens
    # )  # preprocess should be done with trainset

    # # save vocab and datasets
    # _dump_vocab_to_json(vocab_size, token_to_id,
    #                     id_to_token, max_len, file_dir)
