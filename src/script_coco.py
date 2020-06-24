from create_data_files import get_vocab_info, get_dataset, PATH_DATASETS_RSICD, PATH_RSICD
from collections import defaultdict
import spacy
from spacy.tokens import Doc
from preprocess_data.tokens import WhitespaceTokenizer
import inflect
from collections import Counter
import torch
from collections import OrderedDict
from preprocess_data.tokens import convert_captions_to_Y, convert_captions_to_Y_and_POS
import pandas as pd
from datetime import datetime
import json

if __name__ == "__main__":

    raw_dataset = pd.read_json(PATH_RSICD + "raw_dataset/dataset_rsicd.json")

    images = []
    annotations = []
    id_annotation = 0

    images_captions = defaultdict(list)
    for row in raw_dataset["images"]:
        image_name = row["filename"]
        image_id = row["imgid"]
        if row["split"] == "test":

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
                tokens = caption_tokens
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

    with open(PATH_DATASETS_RSICD+"test_coco_format.json", 'w+') as f:
        json.dump(test_coco_format, f, indent=2)
