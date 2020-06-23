import os
import torch
import logging
from args_parser import get_args
from create_data_files import PATH_RSICD, PATH_DATASETS_RSICD, get_vocab_info, get_dataset
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
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_aoa_image import ContinuousAttentionAoAImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_sat_image import ContinuousSATImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_aoanet_image import ContinuousAttentionAoANetImageModel
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
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_product_attribute_embedding_within_imagec import ContinuousAttentionProductAttrEmbeddingWithoutScoreWithinImageCModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_vocab_image import ContinuousAttentionVocabImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_product_vocab_image import ContinuousAttentionProductVocabImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_product_multilevel import ContinuousProductAttentionMultilevelAttrEmbeddingAndRegionsImageModel


from torchvision import transforms
from PIL import Image
from preprocess_data.tokens import START_TOKEN, END_TOKEN
import numpy as np
import operator
from nlgeval import compute_metrics
from models.abtract_model import DecodingType
import json
# from coco_caption.pycocotools.coco import COCO
# from coco_caption.pycocoevalcap.eval import COCOEvalCap

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = '0'


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

    # scores = model.test(test_dataset)

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
    elif args.decodying_type == DecodingType.BIGRAM_PROB_COS.value:
        decoding_method = model.inference_with_bigramprob_and_cos
    else:
        decoding_method = model.inference_with_beamsearch

    list_hipotheses = []
    i = 0
    for values in test_dataset["images"]:

        img_name = values["file_name"]
        img_id = values["id"]

        image_name = PATH_RSICD + \
            "raw_dataset/RSICD_images/" + img_name
        image = Image.open(image_name)
        image = transform(image)
        image = image.unsqueeze(0)

        model.decoder.eval()
        model.encoder.eval()

        text_generated = decoding_method(image, args.n_beam)

        list_hipotheses.append({
            "image_id": img_id,
            "caption": text_generated,
        })

        if args.disable_metrics:
            break

        i += 1
        if i == 10:
            break

    #model.save_sentences(args.decodying_type, args.n_beam, list_hipotheses)

    sentences_path = model.MODEL_DIRECTORY + \
        'evaluation_sentences/' + \
        args.file_name + "_"+args.decodying_type + "_"+str(args.n_beam) + '_coco'  # str(self.args.__dict__)

    with open(sentences_path+'.json', 'w+') as f:
        json.dump(list_hipotheses, f, indent=2)

    # grava um path: "rscid_"
    # coco_eval rscid_

    # coco = COCO(test_dataset)
    # cocoRes = coco.loadRes(sentences_path+'.json')

    # cocoEval = COCOEvalCap(coco, cocoRes)

    # cocoEval.params["image_id"] = cocoRes.getImgIds()
    # cocoEval.evaluate()

    # predicted = {"args": [args.__dict__]}
    # avg_score = cocoEval.eval.items()
    # individual_scores = [eva for eva in cocoEval.evalImgs]
    # for i in range(len(individual_scores)):
    #     predicted[individual_scores[i]["image_id"]] = individual_scores[i]
    # predicted["avg_metrics"] = avg_score

    # model.save_scores(args.decodying_type, args.n_beam, predicted, True)

    # scores_path = model.MODEL_DIRECTORY + \
    #     'evaluation_scores/' + \
    #     args.file_name + "_"+args.decodying_type + "_"+str(args.n_beam) + '_coco'  # str(self.args.__dict__)

    # with open(scores_path+'.json', 'w+') as f:
    #     json.dump(predicted, f, indent=2)
