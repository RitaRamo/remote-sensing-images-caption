import sys
sys.path.append('src/')

from definitions_datasets import PATH_RSICD
import torch
from torchvision import transforms, models
from torch import nn
from data_preprocessing.datasets import ClassificationDataset
from torch.utils.data import DataLoader
from utils.early_stop import EarlyStopping
from utils.optimizer import get_optimizer
import logging
import os
import numpy as np
import time
from efficientnet_pytorch import EfficientNet
from scripts_multilabel_classification.train_model import EfficientEmbeddingsNet
from utils.enums import Datasets
from definitions_datasets import get_dataset_paths

DISABLE_STEPS = False
DATASET = "ucm"

FILE_NAME = "classification_efficientnet_ucm"  # "classification_efficientnet_modifiedrsicd"
FINE_TUNE = True
EFFICIENT_NET = True
PRETRAIN_IMAGE_REGIONS = False
EPOCHS = 300
BATCH_SIZE = 8
EPOCHS_LIMIT_WITHOUT_IMPROVEMENT = 5

NUM_WORKERS = 0
OPTIMIZER_TYPE = "adam"
OPTIMIZER_LR = 1e-4


if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s \nCount %i gpus",
                 device, torch.cuda.device_count())

    if DATASET == Datasets.RSICD.value:
        CLASSIFICATION_DATASET_PATH = "classification_dataset"
    elif DATASET == Datasets.UCM.value:
        CLASSIFICATION_DATASET_PATH = "classification_dataset_ucm"
    else:
        raise Exception("Invalid dataset")

    print("Path of classificaion dataset", CLASSIFICATION_DATASET_PATH)

    dataset_folder, dataset_jsons = get_dataset_paths(DATASET)
    print("dataset folder", dataset_folder)

    classification_state = torch.load(dataset_jsons + CLASSIFICATION_DATASET_PATH)
    classes_to_id = classification_state["classes_to_id"]
    id_to_classes = classification_state["id_to_classes"]
    classification_dataset = classification_state["classification_dataset"]

    dataset_len = len(classification_dataset)
    split_ratio = int(dataset_len * 0.10)

    classification_train = dict(list(classification_dataset.items())[split_ratio:])
    classification_val = dict(list(classification_dataset.items())[0:split_ratio])

    train_dataset_args = (classification_train, dataset_folder + "raw_dataset/images/", classes_to_id)
    val_dataset_args = (classification_val, dataset_folder + "raw_dataset/images/", classes_to_id)

    train_dataloader = DataLoader(
        ClassificationDataset(*train_dataset_args),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_dataloader = DataLoader(
        ClassificationDataset(*val_dataset_args),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    vocab_size = len(classes_to_id)

    # checkpoint =  torch.load('experiments/results/classification_finetune.pth.tar')
    checkpoint = torch.load('experiments/results/' + FILE_NAME + '.pth.tar')
    print("checkpoint loaded")
    if EFFICIENT_NET:
        if PRETRAIN_IMAGE_REGIONS:
            image_model = EfficientEmbeddingsNet()
            #print("image model childern", list(image_model.children())[:-5])
            #num_features = image_model.cnn._fc.in_features
            embedding_size = 300
            image_model.cnn._fc = nn.Linear(embedding_size, vocab_size)
            print("new image model", image_model)
        else:
            image_model = EfficientNet.from_pretrained('efficientnet-b5')
            num_features = image_model._fc.in_features
            image_model._fc = nn.Linear(num_features, vocab_size)
            print("image model loaded")
    else:
        image_model = models.densenet201(pretrained=True)
        num_features = image_model.classifier.in_features
        image_model.classifier = nn.Linear(num_features, vocab_size)

    image_model.load_state_dict(checkpoint['model'])
    image_model.eval()

    def compute_acc(dataset, train_or_val):
        total_acc = torch.tensor([0.0])

        for batch, (img, target) in enumerate(dataset):

            result = image_model(img)
            output = torch.sigmoid(result)

            # print("target", target)
            # preds = output > 0.5
            # text = [id_to_classes[i] for i, value in enumerate(target[0]) if value == 1]
            # output_text = [id_to_classes[i] for i, value in enumerate(preds[0]) if value == True]

            # print("target ext", text)
            # print("output_text ext", output_text)

            condition_1 = (output > 0.5)
            condition_2 = (target == 1)
            correct_preds = torch.sum(condition_1 * condition_2, dim=1)
            n_preds = torch.sum(condition_1, dim=1)

            acc = correct_preds.double() / n_preds
            acc[torch.isnan(acc)] = 0  # n_preds can be 0
            acc_batch = torch.mean(acc)

            total_acc += acc_batch

            # print("corre preds", correct_preds)
            # print("n_preds preds", n_preds)
            # print("acc", acc)

            # print("acc_batch", total_acc)
            # print("total acc", total_acc)
            if batch % 5 == 0:
                # print("n_preds", n_preds)
                # print("acc", acc)
                print("acc_batch", acc_batch.item())
                print("total loss", total_acc)

        print("len of train_data", len(train_dataloader))
        epoch_acc = (total_acc / (batch + 1)).item()
        print("epoch acc", train_or_val, epoch_acc)
        return epoch_acc

    epoch_acc_train = compute_acc(train_dataloader, "TRAIN")
    epoch_acc_val = compute_acc(val_dataloader, "VAL")

    print("train epoch", epoch_acc_train)
    print("val epoch", epoch_acc_val)
