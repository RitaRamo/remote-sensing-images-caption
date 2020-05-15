
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from args_parser import get_args
from create_data_files import (PATH_DATASETS_RSICD, PATH_RSICD, get_dataset,
                               get_vocab_info)
from datasets import CaptionDataset, POSCaptionDataset

from models.basic_encoder_decoder_models.encoder_decoder import BasicEncoderDecoderModel
from models.basic_encoder_decoder_models.encoder_decoder_variants.attention import BasicAttentionModel
from models.basic_encoder_decoder_models.encoder_decoder_variants.mask import BasicMaskGroundTruthWithPredictionModel
from models.basic_encoder_decoder_models.encoder_decoder_variants.sat import BasicShowAttendAndTellModel
from models.continuous_encoder_decoder_models.encoder_decoder import ContinuousEncoderDecoderModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention import ContinuousAttentionModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_relu import ContinuousAttentionReluModel
#from models.continuous_encoder_decoder_models.encoder_decoder_variants.diff_loss import ContinuousMarginModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.bert import ContinuousBertModel
from preprocess_data.images import augment_image
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_schedule_sampling import ContinuousAttentionWithScheduleSamplingModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_schedule_sampling_alt import ContinuousAttentionWithScheduleSamplingAltModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_image import ContinuousAttentionImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_schedule_alt_with_image import ContinuousAttentionImageWithScheduleSamplingModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_image_pos import ContinuousAttentionImagePOSModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_image import ContinuousEncoderDecoderImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_image_and_classification import ContinuousEncoderDecoderImageClassificationModel

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

    vocab_info = get_vocab_info(PATH_DATASETS_RSICD+"vocab_info.json")
    vocab_size, token_to_id, id_to_token, max_len = vocab_info[
        "vocab_size"], vocab_info["token_to_id"], vocab_info["id_to_token"], vocab_info["max_len"]
    logging.info("vocab size %s", vocab_size)

    train_dataset_args = (PATH_DATASETS_RSICD+"train.json",
                          PATH_RSICD+"raw_dataset/RSICD_images/",
                          max_len,
                          token_to_id
                          )

    val_dataset_args = (PATH_DATASETS_RSICD+"val.json",
                        PATH_RSICD+"raw_dataset/RSICD_images/",
                        max_len,
                        token_to_id
                        )

    # if args.augment_data:
    #     transform = transforms.Compose([
    #         # augment_image(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406],  # mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD
    #                              std=[0.229, 0.224, 0.225])
    #     ])

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],  # mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD
    #                             std=[0.229, 0.224, 0.225])
    # ])

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
