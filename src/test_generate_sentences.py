import os
import torch
import logging
from args_parser import get_args
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
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_image_tanh import ContinuousEncoderDecoderImageTanhModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_image_alt import ContinuousEncoderDecoderImageAltModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_image_dist import ContinuousEncoderDecoderImageDistModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_score_tanh_image import ContinuousEncoderDecoderScoreTanhImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_score_tanh_image_tanh import ContinuousEncoderDecoderScoreTanhImageTanhModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_adaptative import ContinuousAdaptativeAttentionImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_adaptative_only import ContinuousAdaptativeAttentionOnlyImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_adaptative_image import ContinuousAdaptativeAttentionImageCompModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_adaptative_drop import ContinuousAdaptativeAttentionDropModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_image_emb import ContinuousEncoderDecoderImageEmbModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_image_distemb import ContinuousEncoderDecoderImageDistEmbModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_adaptative_drop_image import ContinuousAdaptativeAttentionDropImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_2embeddings import ContinuousEncoderDecoder2EmbModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_no_bias import ContinuousEncoderDecoderNoBiasModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_prob_sim import ContinuousEncoderDecoderProbSimModel

from torchvision import transforms
from PIL import Image
from data_preprocessing.preprocess_tokens import START_TOKEN, END_TOKEN
import numpy as np
import operator
from nlgeval import compute_metrics
import json
import cv2

from utils.enums import DecodingType, EvalDatasetType
from definitions_datasets import get_dataset_paths, get_test_path, PATH_EVALUATION_SENTENCES

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = '0'


if __name__ == "__main__":
    device = torch.device("cpu")

    args = get_args()
    print(args.__dict__)

    dataset_folder, dataset_jsons = get_dataset_paths(args.dataset)

    vocab_info = get_vocab_info(dataset_jsons + "vocab_info.json")
    vocab_size, token_to_id, id_to_token, max_len = vocab_info[
        "vocab_size"], vocab_info["token_to_id"], vocab_info["id_to_token"], vocab_info["max_len"]
    print("vocab size", vocab_size)

    test_path, decoding_args = get_test_path(args, dataset_jsons)
    print("test path", test_path)
    test_dataset = get_dataset(test_path)

    model_class = globals()[args.model_class_str]
    model = model_class(
        args, vocab_size, token_to_id, id_to_token, max_len, device)
    model.setup_to_test()

    transform = transforms.Compose([
        transforms.ToTensor(),  # histogram
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
    elif args.decodying_type == DecodingType.BIGRAM_PROB.value:
        decoding_method = model.inference_with_bigramprob
    elif args.decodying_type == DecodingType.BIGRAM_PROB_IMAGE.value:
        decoding_method = model.inference_with_bigramprob_and_image
    elif args.decodying_type == DecodingType.BIGRAM_PROB_COS.value:
        decoding_method = model.inference_with_bigramprob_and_cos
    elif args.decodying_type == DecodingType.BEAM_RANKED_IMAGE.value:
        decoding_method = model.inference_with_beamsearch_ranked_image
    elif args.decodying_type == DecodingType.BEAM_RANKED_BIGRAM.value:
        decoding_method = model.inference_with_beamsearch_ranked_bigram
    elif args.decodying_type == DecodingType.BEAM_TUTORIAL.value:
        decoding_method = model.inference_beam_tutorial
    elif args.decodying_type == DecodingType.BEAM_COMP.value:
        decoding_method = model.inference_beam_comp
    elif args.decodying_type == DecodingType.BEAM_WITHOUT_REFINEMENT.value:
        decoding_method = model.inference_beam_without_refinement
    else:
        print("using beam")
        decoding_method = model.inference_with_beamsearch

    list_hipotheses = []
    i = 0
    for values in test_dataset["images"]:

        img_name = values["file_name"]
        img_id = values["id"]

        image_name = dataset_folder + \
            "raw_dataset/images/" + img_name
        #image = Image.open(image_name)
        image = cv2.imread(image_name)
        image = transform(image)
        image = image.unsqueeze(0)

        # now the eval is inside the model.setup_to_test()
        # model.decoder.eval()
        # model.encoder.eval()

        text_generated = decoding_method(image, args.n_beam, args.min_len, args.rep_window, args.max_len)

        list_hipotheses.append({
            "image_id": img_id,
            "caption": text_generated,
        })

        if args.disable_metrics:
            break

        # i += 1
        # if i == 10:
        #     break

    sentences_path = PATH_EVALUATION_SENTENCES + decoding_args

    with open(sentences_path + '.json', 'w+') as f:
        json.dump(list_hipotheses, f, indent=2)
