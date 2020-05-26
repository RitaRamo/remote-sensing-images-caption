from create_data_files import PATH_RSICD
import torch
from torchvision import transforms, models
from torch import nn
from datasets import ClassificationDataset
from torch.utils.data import DataLoader
from utils.early_stop import EarlyStopping
from optimizer import get_optimizer
import logging
import os
import numpy as np
import time
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F

DISABLE_STEPS = False
FILE_NAME = "classification_efficientnet_focalloss"
FINE_TUNE = True
EFFICIENT_NET = True
FOCAL_LOSS = True
EPOCHS = 300
BATCH_SIZE = 8
EPOCHS_LIMIT_WITHOUT_IMPROVEMENT = 5


NUM_WORKERS = 0
OPTIMIZER_TYPE = "adam"
OPTIMIZER_LR = 1e-4


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class ClassificationModel():
    MODEL_DIRECTORY = "experiments/results/"

    def __init__(self, vocab_size, device):
        self.device = device
        self.checkpoint_exists = False

        if EFFICIENT_NET:
            image_model = EfficientNet.from_pretrained('efficientnet-b4')
            num_features = image_model._fc.in_features
            image_model._fc = nn.Linear(num_features, vocab_size)
        else:  # use densenet
            image_model = models.densenet201(pretrained=True)
            num_features = image_model.classifier.in_features
            image_model.classifier = nn.Linear(num_features, vocab_size)

        self.model = image_model.to(self.device)

    def setup_to_train(self):
        if FOCAL_LOSS:
            self.criterion = FocalLoss().to(self.device)
        else:
            self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        if FINE_TUNE:
            self.optimizer = get_optimizer(OPTIMIZER_TYPE, self.model.parameters(), OPTIMIZER_LR)
        else:  # ConvNet as fixed feature extractor (freeze all the network except the final layer)
            self.optimizer = get_optimizer(OPTIMIZER_TYPE, self.model.classifier.parameters(), OPTIMIZER_LR)

        self._load_weights_from_checkpoint(load_to_train=True)

    def train_step(self, imgs, targets):
        imgs = imgs.to(self.device)
        targets = targets.to(self.device)

        outputs = self.model(imgs)

        loss = self.criterion(outputs, targets)

        self.model.zero_grad()
        loss.backward()

        # Update weights
        self.optimizer.step()

        return loss

    def val_step(self, imgs, targets):
        imgs = imgs.to(self.device)
        targets = targets.to(self.device)

        outputs = self.model(imgs)
        loss = self.criterion(outputs, targets)

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
            epoch_loss = train_total_loss/(batch_i+1)
            logging.info('Time taken for 1 epoch {:.4f} sec'.format(
                time.time() - start))
            logging.info('\n\n-----> TRAIN END! Epoch: {}; Loss: {:.4f}\n'.format(epoch,
                                                                                  train_total_loss/(batch_i+1)))

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
            epoch_val_loss = val_total_loss/(batch_i+1)

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
        path = self.MODEL_DIRECTORY + FILE_NAME+'.pth.tar'
        return path


if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s \nCount %i gpus",
                 device, torch.cuda.device_count())

    classification_state = torch.load("src/data/RSICD/datasets/classification_dataset")
    classes_to_id = classification_state["classes_to_id"]
    id_to_classes = classification_state["id_to_classes"]
    classification_dataset = classification_state["classification_dataset"]

    dataset_len = len(classification_dataset)
    split_ratio = int(dataset_len * 0.10)

    classification_train = dict(list(classification_dataset.items())[split_ratio:])
    classification_val = dict(list(classification_dataset.items())[0:split_ratio])

    train_dataset_args = (classification_train, PATH_RSICD+"raw_dataset/RSICD_images/", classes_to_id)
    val_dataset_args = (classification_val, PATH_RSICD+"raw_dataset/RSICD_images/", classes_to_id)

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

    model = ClassificationModel(vocab_size, device)
    model.setup_to_train()
    model.train(train_dataloader, val_dataloader)

    # checkpoint = torch.load('experiments/results/classification_finetune.pth.tar')

    # image_model = models.densenet201(pretrained=True)
    # num_features = image_model.classifier.in_features
    # image_model.classifier = nn.Linear(num_features, vocab_size)

    # image_model.load_state_dict(checkpoint['model'])

    # for batch, (img, target) in enumerate(train_dataloader):
    #     if batch == 5:
    #         result = image_model(img)
    #         output = torch.sigmoid(result)
    #         output = output > 0.5

    #         print("result", result)
    #         print("output", output)

    #         print("target", target)
    #         text = [id_to_classes[i] for i, value in enumerate(target[0]) if value == 1]
    #         output_text = [id_to_classes[i] for i, value in enumerate(output[0]) if value == True]
    #         print("target ext", text)
    #         print("output_text ext", output_text)

    #         break
