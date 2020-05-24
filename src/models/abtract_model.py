import operator
from preprocess_data.tokens import START_TOKEN, END_TOKEN
from torchvision import transforms
from PIL import Image
from create_data_files import PATH_RSICD
from abc import ABC, abstractmethod
import torch
import json
import os
import logging
import numpy as np
import time
from utils.early_stop import EarlyStopping
from nlgeval import NLGEval
from optimizer import get_optimizer, clip_gradient


class AbstractEncoderDecoderModel(ABC):

    MODEL_DIRECTORY = "experiments/results/"

    def __init__(
        self,
        args,
        vocab_size,
        token_to_id,
        id_to_token,
        max_len,
        device

    ):
        self.args = args
        self.vocab_size = vocab_size
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.max_len = max_len

        self.encoder = None
        self.decoder = None
        self.checkpoint_exists = False
        self.device = device
        self.current_epoch = None

    def setup_to_train(self):
        self._initialize_encoder_and_decoder()
        self._define_optimizers()
        self._load_weights_from_checkpoint(load_to_train=True)
        self._define_loss_criteria()

    def setup_to_test(self):
        self._initialize_encoder_and_decoder()
        self._load_weights_from_checkpoint(load_to_train=False)

    @abstractmethod
    def _initialize_encoder_and_decoder(self):
        pass

    def _define_optimizers(self):
        self.decoder_optimizer = get_optimizer(
            self.args.optimizer_type, self.decoder.parameters(), self.args.decoder_lr)

        self.encoder_optimizer = get_optimizer(
            self.args.optimizer_type,
            self.encoder.parameters(),
            self.args.decoder_lr
        ) if self.args.fine_tune_encoder else None

    @abstractmethod
    def _define_loss_criteria(self):
        pass

    def train(self, train_dataloader, val_dataloader, print_freq):
        early_stopping = EarlyStopping(
            epochs_limit_without_improvement=self.args.epochs_limit_without_improvement,
            epochs_since_last_improvement=self.checkpoint_epochs_since_last_improvement
            if self.checkpoint_exists else 0,
            baseline=self.checkpoint_val_loss if self.checkpoint_exists else np.Inf,
            encoder_optimizer=self.encoder_optimizer,
            decoder_optimizer=self.decoder_optimizer
        )

        start_epoch = self.checkpoint_start_epoch if self.checkpoint_exists else 0

        # Iterate by epoch
        for epoch in range(start_epoch, self.args.epochs):
            self.current_epoch = epoch

            if early_stopping.is_to_stop_training_early():
                break

            start = time.time()
            train_total_loss = 0.0
            val_total_loss = 0.0

            # Train by batch
            self.decoder.train()
            self.encoder.train()
            for batch_i, (imgs, caps, caplens) in enumerate(train_dataloader):

                train_loss = self.train_step(
                    imgs, caps, caplens
                )

                self._log_status("TRAIN", epoch, batch_i,
                                 train_dataloader, train_loss, print_freq)

                train_total_loss += train_loss

                # (only for debug: interrupt val after 1 step)
                if self.args.disable_steps:
                    break

            # End training
            epoch_loss = train_total_loss/(batch_i+1)
            logging.info('Time taken for 1 epoch {:.4f} sec'.format(
                time.time() - start))
            logging.info('\n\n-----> TRAIN END! Epoch: {}; Loss: {:.4f}\n'.format(epoch,
                                                                                  train_total_loss/(batch_i+1)))

            # Start validation
            self.decoder.eval()  # eval mode (no dropout or batchnorm)
            self.encoder.eval()

            with torch.no_grad():

                for batch_i, (imgs, caps, caplens) in enumerate(val_dataloader):

                    val_loss = self.val_step(
                        imgs, caps, caplens)

                    self._log_status("VAL", epoch, batch_i,
                                     val_dataloader, val_loss, print_freq)

                    val_total_loss += val_loss

                    # (only for debug: interrupt val after 1 step)
                    if self.args.disable_steps:
                        break

            # End validation
            epoch_val_loss = val_total_loss/(batch_i+1)

            early_stopping.check_improvement(epoch_val_loss)

            self._save_checkpoint(early_stopping.is_current_val_best(),
                                  epoch,
                                  early_stopping.get_number_of_epochs_without_improvement(),
                                  epoch_val_loss)

            logging.info('\n-------------- END EPOCH:{}⁄{}; Train Loss:{:.4f}; Val Loss:{:.4f} -------------\n'.format(
                epoch, self.args.epochs, epoch_loss, epoch_val_loss))

    def train_step(self, imgs, caps_input, cap_len):
        encoder_out, caps_sorted, caption_lengths = self._prepare_inputs_to_forward_pass(
            imgs, caps_input, cap_len)

        predict_output = self._predict(
            encoder_out, caps_sorted, caption_lengths)
        loss = self._calculate_loss(
            predict_output, caps_sorted, caption_lengths)

        self.decoder_optimizer.zero_grad()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()
        loss.backward()

        # # Clip gradients
        # clip_gradient(self.decoder_optimizer, 5.)
        # if self.encoder_optimizer is not None:
        #     clip_gradient(self.encoder_optimizer, 5.)

        # Update weights
        self.decoder_optimizer.step()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.step()

        return loss

    def val_step(self, imgs, caps_input, cap_len):
        encoder_out, caps_sorted, caption_lengths = self._prepare_inputs_to_forward_pass(
            imgs, caps_input, cap_len)

        predict_output = self._predict(
            encoder_out, caps_sorted, caption_lengths)

        loss = self._calculate_loss(
            predict_output, caps_sorted, caption_lengths)

        return loss

    @abstractmethod
    def _predict(self, predict_output, caps_sorted, caption_lengths):
        pass  # depends on the model

    @abstractmethod
    def _calculate_loss(self, predict_output, caps_sorted, caption_lengths):
        pass  # depends on the model

    def _prepare_inputs_to_forward_pass(self, imgs, caps, caption_lengths):
        imgs = imgs.to(self.device)
        caps = caps.to(self.device)
        caption_lengths = caption_lengths.to(self.device)

        # encoder
        encoder_out = self.encoder(imgs)
        encoder_out = encoder_out.view(
            encoder_out.size(0), -1, encoder_out.size(-1))  # flatten

        # sorted captions
        caption_lengths, sort_ind = caption_lengths.squeeze(
            1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        caps_sorted = caps[sort_ind]

        # input captions must not have "end_token"
        caption_lengths = (caption_lengths - 1).tolist()

        return encoder_out, caps_sorted, caption_lengths

    def _log_status(self, train_or_val, epoch, batch_i, dataloader, loss, print_freq):
        if batch_i % print_freq == 0:
            logging.info(
                "{} - Epoch: [{}/{}]; Batch: [{}/{}]\t Loss: {:.4f}\t".format(
                    train_or_val, epoch, self.args.epochs, batch_i,
                    len(dataloader), loss
                )
            )

    def _save_checkpoint(self, val_loss_improved, epoch, epochs_since_last_improvement, val_loss):
        if val_loss_improved:

            state = {'epoch': epoch,
                     'epochs_since_last_improvement': epochs_since_last_improvement,
                     'val_loss': val_loss,
                     'encoder': self.encoder.state_dict(),
                     'decoder': self.decoder.state_dict(),
                     'encoder_optimizer': self.encoder_optimizer.state_dict() if self.encoder_optimizer else None,
                     'decoder_optimizer': self.decoder_optimizer.state_dict()
                     }
            torch.save(state, self.get_checkpoint_path())

    def _load_weights_from_checkpoint(self, load_to_train):
        checkpoint_path = self.get_checkpoint_path()

        if os.path.exists(checkpoint_path):
            self.checkpoint_exists = True

            checkpoint = torch.load(checkpoint_path)

            # load model weights
            self.decoder.load_state_dict(checkpoint['decoder'])
            self.encoder.load_state_dict(checkpoint['encoder'])

            if load_to_train:
                # load optimizers and start epoch
                self.decoder_optimizer.load_state_dict(
                    checkpoint['decoder_optimizer'])
                if self.encoder_optimizer:
                    self.encoder_optimizer.load_state_dict(
                        checkpoint['encoder_optimizer'])
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
        path = self.MODEL_DIRECTORY + 'trained_models/' + self.args.file_name+'.pth.tar'
        print("get checkpoint path", path)
        return path

    def save_scores(self, decoding_type, scores):
        scores = {key: str(values) for key, values in scores.items()}

        scores_path = self.MODEL_DIRECTORY + \
            'evaluation_scores/' + \
            self.args.file_name + decoding_type  # str(self.args.__dict__)
        with open(scores_path+'.json', 'w+') as f:
            json.dump(scores, f, indent=2)

    @abstractmethod
    def generate_output_index(self, input_word, encoder_out, h, c):
        pass
