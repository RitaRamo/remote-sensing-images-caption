import os
import torch
import logging
from args_parser import get_args
from definitions import PATH_RSICD, PATH_DATASETS_RSICD
from data_preprocessing.create_data_files import get_vocab_info, get_dataset
from models.basic_encoder_decoder_models.encoder_decoder import BasicEncoderDecoderModel
from models.basic_encoder_decoder_models.encoder_decoder_variants.attention import BasicAttentionModel
from models.basic_encoder_decoder_models.encoder_decoder_variants.sat import BasicShowAttendAndTellModel
from models.continuous_encoder_decoder_models.encoder_decoder import ContinuousEncoderDecoderModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention import ContinuousAttentionModel
from models.basic_encoder_decoder_models.encoder_decoder_variants.mask import BasicMaskGroundTruthWithPredictionModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_relu import ContinuousAttentionReluModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.bert import ContinuousBertModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_schedule_sampling import ContinuousAttentionWithScheduleSamplingModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_schedule_sampling_alt import ContinuousAttentionWithScheduleSamplingAltModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_image import ContinuousAttentionImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_schedule_alt_with_image import ContinuousAttentionImageWithScheduleSamplingModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_image_pos import ContinuousAttentionImagePOSModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_image import ContinuousEncoderDecoderImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_image_and_classification import ContinuousEncoderDecoderImageClassificationModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_sat_image import ContinuousSATImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_product_image import ContinuousProductAttentionImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_attribute_softmax_image import ContinuousAttentionAttrSoftmaxImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_attribute_embedding_image import ContinuousAttentionAttrEmbeddingImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_attr_as_image import ContinuousAttentionAttrAsImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_attribute_embedding_withoutscore_image import ContinuousAttentionAttrEmbeddingWithoutScoreImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_attribute_embedding_scorebeforeatt_image import ContinuousAttentionAttrEmbeddingScoreBeforeImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_attribute_embedding_scorecat_image import ContinuousAttentionAttrEmbeddingScoreCatImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_attr_paper import ContinuousAttentionAttrPaperImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_product_image_within_model import ContinuousProductAttentionImageWithinModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_product_imagealt import ContinuousProductAttentionImageAltModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_product_attribute_embedding_image import ContinuousAttentionProductAttrEmbeddingWithoutScoreImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_attribute_embedding_with_regions_image import ContinuousAttentionAttrEmbeddingWithRegionsImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_multilevel_attr_and_regions_image import ContinuousAttentionMultilevelAttrEmbeddingAndRegionsImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_product_attribute_embedding_within_image import ContinuousAttentionProductAttrEmbeddingWithoutScoreWithinImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_out import ContinuousAttentionOutModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_product_attribute_embedding_within_imagec import ContinuousAttentionProductAttrEmbeddingWithoutScoreWithinImageCModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_vocab_image import ContinuousAttentionVocabImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_product_vocab_image import ContinuousAttentionProductVocabImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_product_multilevel import ContinuousProductAttentionMultilevelAttrEmbeddingAndRegionsImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_product_multilevel_1query import ContinuousProductAttentionMultilevelAttrEmbeddingAndRegionsOneQueryImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_product_attribute_embedding_imagec import ContinuousAttentionProductAttrEmbeddingWithoutScoreImageCModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_image_h import ContinuousAttentionImageHModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.neighbour_model import ContinuousNeighbourModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_image_normalized import ContinuousAttentionImageNormalizedModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_image_attr600 import ContinuousAttentionImageAttr600Model
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_image_w import ContinuousEncoderDecoderImageWModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_out import ContinuousEncoderDecoderOutModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_multilevel_region_memory_image import ContinuousAttentionMultilevelRegionMemoryImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_multilevel_region_memory import ContinuousAttentionMultilevelRegionMemoryModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_imagec import ContinuousEncoderDecoderImageCModel
import json

from torchvision import transforms
from PIL import Image
from data_preprocessing.preprocess_tokens import START_TOKEN, END_TOKEN
import numpy as np
import operator
from nlgeval import NLGEval
from models.abtract_model import DecodingType

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = '0'


if __name__ == "__main__":
    device = torch.device("cpu")

    args = get_args()
    print(args.__dict__)

    vocab_info = get_vocab_info(PATH_DATASETS_RSICD + "vocab_info.json")
    vocab_size, token_to_id, id_to_token, max_len = vocab_info[
        "vocab_size"], vocab_info["token_to_id"], vocab_info["id_to_token"], vocab_info["max_len"]
    print("vocab size", vocab_size)

    if args.test_set:
        decoding_args = args.file_name + "_" + args.decodying_type + "_" + str(args.n_beam) + '_coco'
        test_dataset = get_dataset(PATH_DATASETS_RSICD + "test_coco_format.json")
    else:  # validation set
        test_dataset = get_dataset(PATH_DATASETS_RSICD + "val_coco_format.json")
        decoding_args = args.file_name + "_v_" + args.decodying_type + "_" + str(args.n_beam) + '_coco'

    model_class = globals()[args.model_class_str]
    model = model_class(
        args, vocab_size, token_to_id, id_to_token, max_len, device)
    model.setup_to_test()
    #scores = model.test(test_dataset)

    # # start test!
    predicted = {"args": [args.__dict__]}
    metrics = {}

    n_comparations = 0
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD
                             std=[0.229, 0.224, 0.225])
    ])

    # mudar este beam search!
    if args.decodying_type == DecodingType.GREEDY.value:
        decoding_method = model.inference_with_greedy
    elif args.decodying_type == DecodingType.GREEDY_EMBEDDING.value:
        decoding_method = model.inference_with_greedy_embedding

    elif args.decodying_type == DecodingType.GREEDY_SMOOTHL1.value:
        decoding_method = model.inference_with_greedy_smoothl1
    elif args.decodying_type == DecodingType.BEAM_PERPLEXITY.value:
        decoding_method = model.inference_with_perplexity
    elif args.decodying_type == DecodingType.POSTPROCESSING_PERPLEXITY.value:
        decoding_method = model.inference_with_postprocessing_perplexity
    elif args.decodying_type == DecodingType.BIGRAM_PROB.value:
        decoding_method = model.inference_with_bigramprob
    elif args.decodying_type == DecodingType.BIGRAM_PROB_IMAGE.value:
        decoding_method = model.inference_with_bigramprob_and_image
    elif args.decodying_type == DecodingType.POSTPROCESSING_BIGRAM_PROB.value:
        decoding_method = model.inference_with_postprocessing_bigramprob
        print("entrei aqui")
    elif args.decodying_type == DecodingType.BIGRAM_PROB_COS.value:
        decoding_method = model.inference_with_bigramprob_and_cos
    else:
        decoding_method = model.inference_with_beamsearch

    # escolher aqui o id da imagem
    # depois gravar num ficheiro
    # ver os scores do end_token!
    # for img_name, references in test_dataset.items():
    #     image_name = PATH_RSICD + \
    #         "raw_dataset/RSICD_images/" + img_name
    #     image = Image.open(image_name)
    #     image = transform(image)
    #     image = image.unsqueeze(0)

    #     model.decoder.eval()
    #     model.encoder.eval()

    #     text_generated = decoding_method(image, args.n_beam)
    #     break

    images_ids = [1251, 1252, 1260, 1263, 1266, 1269, 1275, 1274, 1277, 1280, 1281, 1287]
    #images_ids = ["1251", "1252", "1260", "1263", "1266", "1269", "1275", "1274", "1277", "1280", "1281"," 1287"]
    all_results = {}
    for values in test_dataset["images"]:

        img_name = values["file_name"]
        img_id = values["id"]
        #print("img_id", type(img_id))
        # break

        if img_id in images_ids:
            print("entrei aqui")

            image_name = PATH_RSICD + \
                "raw_dataset/RSICD_images/" + img_name
            image = Image.open(image_name)
            image = transform(image)
            image = image.unsqueeze(0)

            # know the eval is inside the model.setup_to_test()
            # model.decoder.eval()
            # model.encoder.eval()

            text_generated, beam_results = decoding_method(image, args.n_beam)
            all_results[img_id] = beam_results

    with open("beam_results.json", 'w+') as f:
        json.dump(all_results, f, indent=2)
