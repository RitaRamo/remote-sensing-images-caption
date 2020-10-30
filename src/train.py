
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from args_parser import get_args
from data_preprocessing.create_data_files import get_dataset, get_vocab_info
from data_preprocessing.datasets import CaptionDataset, POSCaptionDataset

from models.basic_encoder_decoder_models.encoder_decoder import BasicEncoderDecoderModel
from models.basic_encoder_decoder_models.encoder_decoder_variants.attention import BasicAttentionModel
from models.basic_encoder_decoder_models.encoder_decoder_variants.mask import BasicMaskGroundTruthWithPredictionModel
from models.basic_encoder_decoder_models.encoder_decoder_variants.sat import BasicShowAttendAndTellModel
from models.continuous_encoder_decoder_models.encoder_decoder import ContinuousEncoderDecoderModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention import ContinuousAttentionModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_relu import ContinuousAttentionReluModel
#from models.continuous_encoder_decoder_models.encoder_decoder_variants.diff_loss import ContinuousMarginModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.bert import ContinuousBertModel
from data_preprocessing.preprocess_images import augment_image
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
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_attr_paper import ContinuousAttentionAttrPaperImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_attribute_embedding_image import ContinuousAttentionAttrEmbeddingImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_attr_as_image import ContinuousAttentionAttrAsImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_attribute_embedding_withoutscore_image import ContinuousAttentionAttrEmbeddingWithoutScoreImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_attribute_embedding_scorebeforeatt_image import ContinuousAttentionAttrEmbeddingScoreBeforeImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_attribute_embedding_scorecat_image import ContinuousAttentionAttrEmbeddingScoreCatImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_product_image_within_model import ContinuousProductAttentionImageWithinModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_product_imagealt import ContinuousProductAttentionImageAltModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_product_attribute_embedding_image import ContinuousAttentionProductAttrEmbeddingWithoutScoreImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_attribute_embedding_with_regions_image import ContinuousAttentionAttrEmbeddingWithRegionsImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_out import ContinuousAttentionOutModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_multilevel_attr_and_regions_image import ContinuousAttentionMultilevelAttrEmbeddingAndRegionsImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_product_attribute_embedding_within_image import ContinuousAttentionProductAttrEmbeddingWithoutScoreWithinImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_product_attribute_embedding_within_imagec import ContinuousAttentionProductAttrEmbeddingWithoutScoreWithinImageCModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_vocab_image import ContinuousAttentionVocabImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_product_vocab_image import ContinuousAttentionProductVocabImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_product_multilevel import ContinuousProductAttentionMultilevelAttrEmbeddingAndRegionsImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_product_multilevel_1query import ContinuousProductAttentionMultilevelAttrEmbeddingAndRegionsOneQueryImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_product_attribute_embedding_imagec import ContinuousAttentionProductAttrEmbeddingWithoutScoreImageCModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_image_h import ContinuousAttentionImageHModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_image_normalized import ContinuousAttentionImageNormalizedModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_image_attr600 import ContinuousAttentionImageAttr600Model
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_image_w import ContinuousEncoderDecoderImageWModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_out import ContinuousEncoderDecoderOutModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_multilevel_region_memory_image import ContinuousAttentionMultilevelRegionMemoryImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_multilevel_region_memory_image import ContinuousAttentionMultilevelRegionMemoryImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_multilevel_region_memory import ContinuousAttentionMultilevelRegionMemoryModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_imagec import ContinuousEncoderDecoderImageCModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_image_tanh import ContinuousEncoderDecoderImageTanhModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_image_alt import ContinuousEncoderDecoderImageAltModel
#from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_image_altc import ContinuousEncoderDecoderImageAltCModel
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
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_prob_sim2 import ContinuousEncoderDecoderProbSim2Model
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_concembeddings import ContinuousEncoderDecoderConcEmbModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_probk import ContinuousEncoderDecoderProbSimKModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_prob_simH import ContinuousEncoderDecoderProbSimHModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_prob_sim_only import ContinuousEncoderDecoderProbSimOnlyModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_2layers import ContinuousEncoderDecoder2LayersModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_2layers_after import ContinuousEncoderDecoder2LayersAfterModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_2layers_after_sim_only import ContinuousEncoderDecoder2LayersAfterSimOnlyModel

from definitions_datasets import get_dataset_paths

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# PYTHONHASHSEED=0 python3 src/train.py @experiments/conf_files/<name_of_file>.txt

# apontar que

# nn.Softmax + torch.log + nn.NLLLoss -> might be numerically unstable
# nn.LogSoftmax + nn.NLLLoss -> is perfectly fine for training; to get probabilities you would have to call torch.exp on the output
# usas exp para calcular o log e teres o sotmax(as probs)
# raw logits + nn.CrossEntropyLoss -> also perfectly fine as it calls the second approach internally; to get probabilities you would have to call torch.softmax on the output
# aplicas o softmax paa ter as probs, dado só internmntna loss se ter usado softmax...
if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)

    args = get_args()
    logging.info(args.__dict__)

    if args.set_cpu_device:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s \nCount %i gpus",
                 device, torch.cuda.device_count())

    dataset_folder, dataset_jsons = get_dataset_paths(args.dataset)
    logging.info("dataset folder %s", dataset_folder)
    logging.info("dataset jsons %s", dataset_jsons)

    vocab_info = get_vocab_info(dataset_jsons + "vocab_info.json")
    vocab_size, token_to_id, id_to_token, max_len = vocab_info[
        "vocab_size"], vocab_info["token_to_id"], vocab_info["id_to_token"], vocab_info["max_len"]
    logging.info("vocab size %s", vocab_size)

    train_dataset_args = (dataset_jsons + "train.json",
                          dataset_folder + "raw_dataset/images/",
                          max_len,
                          token_to_id
                          )

    val_dataset_args = (dataset_jsons + "val.json",
                        dataset_folder + "raw_dataset/images/",
                        max_len,
                        token_to_id
                        )

    # vocab_info = get_vocab_info(PATH_DATASETS_RSICD_NEW_TRAIN_AND_VAL + "vocab_info.json")
    # vocab_size, token_to_id, id_to_token, max_len = vocab_info[
    #     "vocab_size"], vocab_info["token_to_id"], vocab_info["id_to_token"], vocab_info["max_len"]
    # logging.info("vocab size %s", vocab_size)

    # train_dataset_args = (PATH_DATASETS_RSICD_NEW_TRAIN_AND_VAL + "train.json",
    #                       PATH_RSICD + "raw_dataset/RSICD_images/",
    #                       max_len,
    #                       token_to_id
    #                       )

    # val_dataset_args = (PATH_DATASETS_RSICD_NEW_TRAIN_AND_VAL + "val.json",
    #                     PATH_RSICD + "raw_dataset/RSICD_images/",
    #                     max_len,
    #                     token_to_id
    #                     )

    if args.pos_tag_dataset:
        train_dataloader = DataLoader(
            POSCaptionDataset(*train_dataset_args, args.augment_data),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        val_dataloader = DataLoader(
            POSCaptionDataset(*val_dataset_args, args.augment_data),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

    else:
        train_dataloader = DataLoader(
            CaptionDataset(*train_dataset_args, args.augment_data),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        val_dataloader = DataLoader(
            CaptionDataset(*val_dataset_args, args.augment_data),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

    model_class = globals()[args.model_class_str]

    model = model_class(
        args, vocab_size, token_to_id, id_to_token, max_len, device)
    model.setup_to_train()
    model.train(train_dataloader, val_dataloader, args.print_freq)
