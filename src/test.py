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


from torchvision import transforms
from PIL import Image
from preprocess_data.tokens import START_TOKEN, END_TOKEN
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

    vocab_info = get_vocab_info(PATH_DATASETS_RSICD+"vocab_info.json")
    vocab_size, token_to_id, id_to_token, max_len = vocab_info[
        "vocab_size"], vocab_info["token_to_id"], vocab_info["id_to_token"], vocab_info["max_len"]
    print("vocab size", vocab_size)

    test_dataset = get_dataset(PATH_DATASETS_RSICD+"test.json")

    model_class = globals()[args.model_class_str]
    model = model_class(
        args, vocab_size, token_to_id, id_to_token, max_len, device)
    model.setup_to_test()

    #scores = model.test(test_dataset)

    # # start test!
    predicted = {"args": [args.__dict__]}
    metrics = {}

    if args.disable_metrics:
        print(
            "disable_metrics = True, thus will not compute metrics")
    else:
        nlgeval = NLGEval()  # loads the metrics models
        print("vou começar")

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
    elif args.decodying_type == DecodingType.BIGRAM_PROB_COS.value:
        decoding_method = model.inference_with_bigramprob_and_cos
    else:
        decoding_method = model.inference_with_beamsearch

    for img_name, references in test_dataset.items():

        image_name = PATH_RSICD + \
            "raw_dataset/RSICD_images/" + img_name
        image = Image.open(image_name)
        image = transform(image)
        image = image.unsqueeze(0)

        model.decoder.eval()
        model.encoder.eval()

        text_generated = decoding_method(image, args.n_beam)

        if args.disable_metrics:
            break

        # TODO:remove metrics that you will not use...
        all_scores = nlgeval.compute_individual_metrics(
            references, text_generated)

        if n_comparations % args.print_freq == 0:
            print("this are dic metrics", all_scores)

        predicted[img_name] = {
            "value": text_generated,
            "scores": all_scores
        }

        for metric, score in all_scores.items():
            if metric not in metrics:
                metrics[metric] = score
            else:
                metrics[metric] += score
        n_comparations += 1

    avg_metrics = {metric: total_score /
                   n_comparations for metric, total_score in metrics.items()
                   }

    predicted['avg_metrics'] = {
        "value": "",
        "scores": avg_metrics
    }

    print("avg_metrics", avg_metrics)

    model.save_scores(args.decodying_type, args.n_beam, predicted)
