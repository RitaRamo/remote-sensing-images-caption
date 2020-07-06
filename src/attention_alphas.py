from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_attribute_embedding_withoutscore_image import ContinuousAttentionAttrEmbeddingWithoutScoreImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_product_attribute_embedding_image import ContinuousAttentionProductAttrEmbeddingWithoutScoreImageModel

import os
import torch
import logging
from args_parser import get_args
from definitions import PATH_RSICD, PATH_DATASETS_RSICD
from create_data_files import get_vocab_info, get_dataset
from torchvision import transforms
from PIL import Image
from preprocess_data.tokens import START_TOKEN, END_TOKEN
import numpy as np
import operator
from nlgeval import NLGEval
from models.abtract_model import DecodingType
from definitions import PATH_DATASETS_RSICD, PATH_RSICD, PATH_EVALUATION_SENTENCES
import json

if __name__ == "__main__":
    device = torch.device("cpu")

    args = get_args()
    print(args.__dict__)

    vocab_info = get_vocab_info(PATH_DATASETS_RSICD+"vocab_info.json")
    vocab_size, token_to_id, id_to_token, max_len = vocab_info[
        "vocab_size"], vocab_info["token_to_id"], vocab_info["id_to_token"], vocab_info["max_len"]
    print("vocab size", vocab_size)

    test_dataset = get_dataset(PATH_DATASETS_RSICD+"test_coco_format.json")

    model_class = globals()[args.model_class_str]
    model = model_class(
        args, vocab_size, token_to_id, id_to_token, max_len, device)
    model.setup_to_test()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD
                             std=[0.229, 0.224, 0.225])
    ])

    list_hipotheses = []
    i = 0

    # img_names = ["00722.jpg", "storagetanks_8.jpg", "storagetanks_81.jpg",
    #              "storagetanks_90.jpg", "forest_88.jpg", "bareland_91.jpg"]
    # img_ids = [10721, 9558, 9560, 9570, 3817, 661]
    img_names = ['airport_62.jpg',
                 'bareland_80.jpg',
                 'baseballfield_9.jpg',
                 'beach_69.jpg',
                 'center_84.jpg',
                 'church_78.jpg',
                 'commercial_76.jpg',
                 'denseresidential_78.jpg',
                 'desert_81.jpg',
                 'farmland_70.jpg',
                 'forest_85.jpg',
                 'industrial_89.jpg',
                 'meadow_97.jpg',
                 'mediumresidential_73.jpg',
                 'mountain_80.jpg',
                 'park_76.jpg',
                 'playground_76.jpg',
                 'pond_84.jpg',
                 'port_90.jpg',
                 'railwaystation_9.jpg',
                 'resort_93.jpg',
                 'river_68.jpg',
                 'school_81.jpg',
                 'sparseresidential_82.jpg',
                 'square_80.jpg',
                 'stadium_74.jpg',
                 'storagetanks_69.jpg',
                 'viaduct_66.jpg']
    img_ids = ['319',
               '649',
               '879',
               '1256',
               '1893',
               '2126',
               '2474',
               '2886',
               '3190',
               '3548',
               '3814',
               '4208',
               '4497',
               '4761',
               '5109',
               '5454',
               '6214',
               '6643',
               '7030',
               '7289',
               '7583',
               '7965',
               '8280',
               '8581',
               '8909',
               '9192',
               '9546',
               '9963']
    # Escolheres 3 em que tens máx score
    # Escolheres 3 em que tens péssimo score
    # for values in test_dataset["images"]:

    for j in range(len(img_names)):

        img_name = img_names[j]
        img_id = img_ids[j]

        image_name = PATH_RSICD + "raw_dataset/RSICD_images/" + img_name
        image = Image.open(image_name)
        image = transform(image)
        image = image.unsqueeze(0)

        model.decoder.eval()
        model.encoder.eval()

        # TODO: tens de mudar o decoding method para considerar o alpha!
        text_generated, alphas, all_similar_embeddings = model.greedy_with_attention(image, args.n_beam)

        list_hipotheses.append({
            "image_id": img_id,
            "caption": text_generated,
            "alphas": alphas,
            "all_similar_embeddings": all_similar_embeddings
        })

        if args.disable_metrics:
            break

        i += 1
        if i == 11:
            break

    sentences_path = PATH_EVALUATION_SENTENCES + \
        args.file_name + "_"+args.decodying_type + "_"+str(args.n_beam) + '_alphas'  # str(self.args.__dict__)

    # with open(sentences_path+'.json', 'w+') as f:
    #     json.dump(list_hipotheses, f, indent=2)

    state = {
        "list_hipotheses": list_hipotheses,
    }

    torch.save(state, sentences_path + ".json")
