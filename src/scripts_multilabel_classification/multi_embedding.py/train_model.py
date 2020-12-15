import sys
sys.path.append('src/')

from definitions_datasets import PATH_RSICD
import torch
from torchvision import transforms, models
from torch import nn
from data_preprocessing.create_data_files import get_dataset, get_vocab_info
from data_preprocessing.datasets import ClassificationEmbeddingDataset, ClassificationCaptionEmbeddingDataset
from torch.utils.data import DataLoader
from utils.early_stop import EarlyStopping
from utils.optimizer import get_optimizer
import logging
import os
import numpy as np
import time
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
from utils.enums import Datasets
from definitions_datasets import get_dataset_paths
from embeddings.embeddings import get_embedding_layer

DISABLE_STEPS = False
#FILE_NAME = "classification_efficientnet_focalloss"
FILE_NAME = "classification_efficientnet_"
DATASET = "ucm"
DATASET_TYPE = "caption"  # caption
TYPE_OF_MULTIMODAL = "embedding"  # sigmoid
LOSS_TYPE= "smoothl1"
FINE_TUNE = True
EFFICIENT_NET = True
EMBED_DIM = 300
EMBEDDING_TYPE="glove"
EPOCHS = 300
BATCH_SIZE = 8
EPOCHS_LIMIT_WITHOUT_IMPROVEMENT = 5


NUM_WORKERS = 0
OPTIMIZER_TYPE = "adam"
OPTIMIZER_LR = 1e-4


class ClassificationModel():
    MODEL_DIRECTORY = "experiments/results/"

    def __init__(self, device):
        self.device = device
        self.checkpoint_exists = False

        if EFFICIENT_NET:
            image_model = EfficientNet.from_pretrained('efficientnet-b5')
            num_features = image_model._fc.in_features
            image_model._fc = nn.Linear(num_features, EMBED_DIM)
            #print("image model", image_model)
        else:  # use densenet
            image_model = models.densenet201(pretrained=True)
            #print("list image model", list(image_model.children()))
            num_features = image_model.classifier.in_features
            image_model.classifier = nn.Linear(num_features, EMBED_DIM)

        self.model = image_model.to(self.device)
        # print(stop)

    def setup_to_train(self):
        #TODO: CHANGE
        if LOSS_TYPE == "smoothl1":
            self.criterion = nn.SmoothL1Loss(reduction='none').to(self.device)
        else:
            self.criterion = nn.CosineEmbeddingLoss().to(self.device)


        if FINE_TUNE:
            self.optimizer = get_optimizer(OPTIMIZER_TYPE, self.model.parameters(), OPTIMIZER_LR)
        else:  # ConvNet as fixed feature extractor (freeze all the network except the final layer)
            self.optimizer = get_optimizer(OPTIMIZER_TYPE, self.model.classifier.parameters(), OPTIMIZER_LR)

        self._load_weights_from_checkpoint(load_to_train=True)

    def train_step(self, imgs, targets):
        imgs = imgs.to(self.device)
        targets = targets.to(self.device)
        outputs = self.model(imgs)

        if LOSS_TYPE == "smoothl1":
            outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1)
            targets = torch.nn.functional.normalize(targets, p=2, dim=-1)
            loss = self.criterion(outputs, targets)
            loss = torch.sum(loss, dim=-1)
            loss = torch.mean(loss)
        else:
            y = torch.ones(outputs.shape[0]).to(self.device)
            loss = self.criterion(outputs, targets, y)

        self.model.zero_grad()
        loss.backward()

        # Update weights
        self.optimizer.step()

        return loss

    def val_step(self, imgs, targets):
        imgs = imgs.to(self.device)
        targets = targets.to(self.device)
        outputs = self.model(imgs)

        if LOSS_TYPE == "smoothl1":
            outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1)
            targets = torch.nn.functional.normalize(targets, p=2, dim=-1)
            loss = self.criterion(outputs, targets)
            loss = torch.sum(loss, dim=-1)
            loss = torch.mean(loss)
        else:
            y = torch.ones(outputs.shape[0]).to(self.device)
            loss = self.criterion(outputs, targets, y)

        return loss

    def train(self, train_dataloader, val_dataloader, print_freq=5):
        early_stopping = EarlyStopping(
            epochs_limit_without_improvement=EPOCHS_LIMIT_WITHOUT_IMPROVEMENT,
            epochs_since_last_improvement=self.checkpoint_epochs_since_last_improvement
            if self.checkpoint_exists else 0,
            baseline=self.checkpoint_val_loss if self.checkpoint_exists else np.Inf,
            encoder_optimizer=None,  # TENS
            decoder_optimizer=None,
            period_decay_lr=1000  # no decay lr!
        )

        start_epoch = self.checkpoint_start_epoch if self.checkpoint_exists else 0

        # Iterate by epoch
        for epoch in range(start_epoch, EPOCHS):
            self.current_epoch = epoch

            if early_stopping.is_to_stop_training_early():
                break

            start = time.time()
            train_total_loss = 0.0
            val_total_loss = 0.0

            # Train by batch
            self.model.train()

            for batch_i, (imgs, targets) in enumerate(train_dataloader):

                train_loss = self.train_step(
                    imgs, targets
                )

                self._log_status("TRAIN", epoch, batch_i,
                                 train_dataloader, train_loss, print_freq)

                train_total_loss += train_loss

                # (only for debug: interrupt val after 1 step)
                if DISABLE_STEPS:
                    break

            # End training
            epoch_loss = train_total_loss / (batch_i + 1)
            logging.info('Time taken for 1 epoch {:.4f} sec'.format(
                time.time() - start))
            logging.info('\n\n-----> TRAIN END! Epoch: {}; Loss: {:.4f}\n'.format(epoch,
                                                                                  train_total_loss / (batch_i + 1)))

            # Start validation
            self.model.eval()  # eval mode (no dropout or batchnorm)

            with torch.no_grad():

                for batch_i, (imgs, targets) in enumerate(val_dataloader):

                    val_loss = self.val_step(
                        imgs, targets)

                    self._log_status("VAL", epoch, batch_i,
                                     val_dataloader, val_loss, print_freq)

                    val_total_loss += val_loss

                    # (only for debug: interrupt val after 1 step)
                    if DISABLE_STEPS:
                        break

            # End validation
            epoch_val_loss = val_total_loss / (batch_i + 1)

            early_stopping.check_improvement(epoch_val_loss)

            self._save_checkpoint(early_stopping.is_current_val_best(),
                                  epoch,
                                  early_stopping.get_number_of_epochs_without_improvement(),
                                  epoch_val_loss)

            logging.info('\n-------------- END EPOCH:{}⁄{}; Train Loss:{:.4f}; Val Loss:{:.4f} -------------\n'.format(
                epoch, EPOCHS, epoch_loss, epoch_val_loss))

    def _log_status(self, train_or_val, epoch, batch_i, dataloader, loss, print_freq):
        if batch_i % print_freq == 0:
            logging.info(
                "{} - Epoch: [{}/{}]; Batch: [{}/{}]\t Loss: {:.4f}\t".format(
                    train_or_val, epoch, EPOCHS, batch_i,
                    len(dataloader), loss
                )
            )

    def _save_checkpoint(self, val_loss_improved, epoch, epochs_since_last_improvement, val_loss):
        if val_loss_improved:

            state = {'epoch': epoch,
                     'epochs_since_last_improvement': epochs_since_last_improvement,
                     'val_loss': val_loss,
                     'model': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict()
                     }
            torch.save(state, self.get_checkpoint_path())

    def _load_weights_from_checkpoint(self, load_to_train):
        checkpoint_path = self.get_checkpoint_path()

        if os.path.exists(checkpoint_path):
            self.checkpoint_exists = True

            checkpoint = torch.load(checkpoint_path)

            # load model weights
            self.model.load_state_dict(checkpoint['model'])

            if load_to_train:
                # load optimizers and start epoch
                self.optimizer.load_state_dict(
                    checkpoint['optimizer'])

                self.checkpoint_start_epoch = checkpoint['epoch'] + 1
                self.checkpoint_epochs_since_last_improvement = checkpoint[
                    'epochs_since_last_improvement'] + 1
                self.checkpoint_val_loss = checkpoint['val_loss']

                logging.info(
                    "Restore model from checkpoint. Start epoch %s ", self.checkpoint_start_epoch)
        else:
            logging.info(
                "No checkpoint. Will start model from beggining\n")

    def get_checkpoint_path(self):
        if DATASET_TYPE == "caption":
            path = self.MODEL_DIRECTORY + FILE_NAME + DATASET +  '_embedding_caption_'+EMBEDDING_TYPE+'_'+LOSS_TYPE+'.pth.tar'
        else:
            path = self.MODEL_DIRECTORY + FILE_NAME + DATASET + '_embedding_nouns_adjs_'+EMBEDDING_TYPE+'_'+LOSS_TYPE+'.pth.tar'
        return path


if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s \nCount %i gpus",
                 device, torch.cuda.device_count())

    if DATASET == Datasets.RSICD.value:
        CLASSIFICATION_DATASET_PATH = "classification_dataset_rsicd"
    elif DATASET == Datasets.UCM.value:
        CLASSIFICATION_DATASET_PATH = "classification_dataset_ucm"
    elif DATASET == Datasets.FLICKR8K.value:
        CLASSIFICATION_DATASET_PATH = "classification_dataset_flickr8k"
    else:
        raise Exception("Invalid dataset")

    if DATASET_TYPE == "caption":
        CLASSIFICATION_DATASET_PATH = CLASSIFICATION_DATASET_PATH + "_caption"
    else:
        CLASSIFICATION_DATASET_PATH = CLASSIFICATION_DATASET_PATH + "_nouns_adjs"

    print("Path of classificaion dataset", DATASET)

    dataset_folder, dataset_jsons = get_dataset_paths(DATASET)
    print("dataset folder", dataset_folder)

    classification_state = torch.load(dataset_jsons + CLASSIFICATION_DATASET_PATH)
    if DATASET_TYPE != "caption":
        classes_to_id = classification_state["classes_to_id"]
        id_to_classes = classification_state["id_to_classes"]
        classid_to_wordid = classification_state["classid_to_wordid"]
    classification_dataset = classification_state["classification_dataset"]

    dataset_len = len(classification_dataset)
    split_ratio = int(dataset_len * 0.10)

    classification_train = dict(list(classification_dataset.items())[split_ratio:])
    classification_val = dict(list(classification_dataset.items())[0:split_ratio])

    vocab_info = get_vocab_info(dataset_jsons + "vocab_info.json")
    vocab_size, token_to_id, id_to_token, max_len = vocab_info[
        "vocab_size"], vocab_info["token_to_id"], vocab_info["id_to_token"], vocab_info["max_len"]
    embedding_matrix = get_embedding_layer(EMBEDDING_TYPE, EMBED_DIM, vocab_size, token_to_id, False)
    # adicionar aqui coisas classtoword id

    if DATASET_TYPE == "caption":
        train_dataset_args = (classification_train, dataset_folder + "raw_dataset/images/",
                              embedding_matrix)
        val_dataset_args = (classification_val, dataset_folder + "raw_dataset/images/",
                            embedding_matrix)

        train_dataloader = DataLoader(
            ClassificationCaptionEmbeddingDataset(*train_dataset_args),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS
        )

        val_dataloader = DataLoader(
            ClassificationCaptionEmbeddingDataset(*val_dataset_args),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS
        )

    else:
        train_dataset_args = (classification_train, dataset_folder + "raw_dataset/images/",
                              classes_to_id, classid_to_wordid, embedding_matrix)
        val_dataset_args = (classification_val, dataset_folder + "raw_dataset/images/",
                            classes_to_id, classid_to_wordid, embedding_matrix)

        train_dataloader = DataLoader(
            ClassificationEmbeddingDataset(*train_dataset_args),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS
        )

        val_dataloader = DataLoader(
            ClassificationEmbeddingDataset(*val_dataset_args),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS
        )

    #TODO: CHANGE
    model = ClassificationModel(device)
    model.setup_to_train()
    model.train(train_dataloader, val_dataloader)
