import operator
from data_preprocessing.preprocess_tokens import START_TOKEN, END_TOKEN
from torchvision import transforms
from PIL import Image
from abc import ABC, abstractmethod
import torch
import json
import os
import logging
import numpy as np
import time
from utils.early_stop import EarlyStopping
from nlgeval import NLGEval
from utils.optimizer import get_optimizer, clip_gradient
# from enum import Enum
from utils.enums import DecodingType
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
import math
from definitions_datasets import PATH_TRAINED_MODELS, PATH_EVALUATION_SCORES


class AbstractEncoderDecoderModel(ABC):

    # MODEL_DIRECTORY = "experiments/results/"

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
        self.decoder.eval()
        self.encoder.eval()

        if self.args.decodying_type == DecodingType.BEAM_PERPLEXITY.value:
            self.language_model_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
            self.language_model = GPT2LMHeadModel.from_pretrained('gpt2-xl')

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
            epoch_loss = train_total_loss / (batch_i + 1)
            logging.info('Time taken for 1 epoch {:.4f} sec'.format(
                time.time() - start))
            logging.info('\n\n-----> TRAIN END! Epoch: {}; Loss: {:.4f}\n'.format(epoch,
                                                                                  train_total_loss / (batch_i + 1)))

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
            epoch_val_loss = val_total_loss / (batch_i + 1)

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
        if load_to_train and self.args.checkpoint_model is not None:
            # get checkpint path of a given model, that you want to keep training with different fine-tuning,etc
            # ex: get model "enc_dec without fine-tuning", then train again with fine-tuning, saving in a new model "enc_dec with fine-tuning"
            checkpoint_path = PATH_TRAINED_MODELS + self.args.checkpoint_model + '.pth.tar'
            checkpoint = torch.load(checkpoint_path)
            self.decoder.load_state_dict(checkpoint['decoder'])
            self.encoder.load_state_dict(checkpoint['encoder'])
            print("load checkpoint of a different model, which will be trained and saved in a new file")

        else:  # get path of current args.model
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
        path = PATH_TRAINED_MODELS + self.args.file_name + '.pth.tar'
        print("get checkpoint path", path)
        return path

    def save_scores(self, decoding_type, n_beam, scores):
        scores = {key: str(values) for key, values in scores.items()}

        scores_path = PATH_EVALUATION_SCORES + \
            self.args.file_name + decoding_type + str(n_beam) + "_startent"  # str(self.args.__dict__)
        with open(scores_path + '.json', 'w+') as f:
            json.dump(scores, f, indent=2)

    def inference_with_greedy(self, image, n_solutions=0, min_len=0):
        with torch.no_grad():  # no need to track history

            decoder_sentence = []

            input_word = torch.tensor([self.token_to_id[START_TOKEN]])

            i = 1

            encoder_output = self.encoder(image)
            encoder_output = encoder_output.view(
                1, -1, encoder_output.size()[-1])

            h, c = self.decoder.init_hidden_state(encoder_output)

            while True:

                scores, h, c = self.generate_output_index(
                    input_word, encoder_output, h, c)

                sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)

                current_output_index = sorted_indices[0]

                current_output_token = self.id_to_token[current_output_index.item(
                )]

                decoder_sentence.append(current_output_token)

                if current_output_token == END_TOKEN:
                    # ignore end_token
                    decoder_sentence = decoder_sentence[:-1]
                    break

                if i >= self.max_len - 1:  # until 35
                    break

                input_word[0] = current_output_index.item()

                i += 1

            generated_sentence = " ".join(decoder_sentence)
            print("\ngenerated sentence:", generated_sentence)
            print(stop)

            return generated_sentence  # input_caption

    def inference_with_beamsearch(self, image, n_solutions=3, min_len=2, repetitions_window=0):

        def compute_probability(seed_text, seed_prob, sorted_scores, index, current_text):
            # return (seed_prob * (len(seed_text)**0.75) + np.log(sorted_scores[index].item())) / ((len(seed_text) + 1)**0.75)
            # print("\nseed_text", seed_text)
            # print("sorted index", sorted_scores[index].item())
            # print("np log index", np.log(sorted_scores[index].item()))
            # print("final", (seed_prob * len(seed_text) + np.log(sorted_scores[index].item())) / (len(seed_text) + 1))

            return (seed_prob * len(seed_text) + np.log(sorted_scores[index].item())) / (len(seed_text) + 1)

        def generate_n_solutions(seed_text, seed_prob, encoder_out, h, c, n_solutions):
            last_token = seed_text[-1]

            if last_token == END_TOKEN:
                if len(seed_text) <= min_len:
                    return [(seed_text, -np.inf, h, c)]
                return [(seed_text, seed_prob, h, c)]

            top_solutions = []
            scores, h, c = self.generate_output_index(
                torch.tensor([self.token_to_id[last_token]]), encoder_out, h, c)

            sorted_scores, sorted_indices = torch.sort(
                scores, descending=True, dim=-1)

            for index in range(n_solutions):
                text = seed_text + [self.id_to_token[sorted_indices[index].item()]]
                # beam search taking into account lenght of sentence
                # prob = (seed_prob*len(seed_text) + np.log(sorted_scores[index].item()) / (len(seed_text)+1))
                text_score = compute_probability(seed_text, seed_prob, sorted_scores, index, text)
                top_solutions.append((text, text_score, h, c))

            return top_solutions

        def get_most_probable(candidates, n_solutions):
            return sorted(candidates, key=operator.itemgetter(1), reverse=True)[:n_solutions]

        with torch.no_grad():
            #my_dict = {}

            encoder_output = self.encoder(image)
            encoder_output = encoder_output.view(1, -1, encoder_output.size()[-1])  # flatten encoder
            h, c = self.decoder.init_hidden_state(encoder_output)

            top_solutions = [([START_TOKEN], 0.0, h, c)]

            for time_step in range(self.max_len):
                candidates = []
                for sentence, prob, h, c in top_solutions:
                    candidates.extend(generate_n_solutions(
                        sentence, prob, encoder_output, h, c, n_solutions))

                top_solutions = get_most_probable(candidates, n_solutions)

                # print("\nall candidates", [(text, prob) for text, prob, _, _ in candidates])
                # # my_dict["cand"].append([(text, prob) for text, prob, _, _ in candidates])
                # print("\ntop", [(text, prob)
                #                 for text, prob, _, _ in top_solutions])
                # my_dict["top"].append([(text, prob) for text, prob, _, _ in top_solutions])
            #     my_dict[time_step] = {"cand": [(text, prob) for text, prob, _, _ in candidates],
            #                           "top": [(text, prob) for text, prob, _, _ in top_solutions]}

            # with open("beam_outro.json", 'w+') as f:
            #     json.dump(my_dict, f, indent=2)
            #     print(stop)
            # print("top solutions", [(text, prob)
            #                         for text, prob, _, _ in top_solutions])
            best_tokens, prob, h, c = top_solutions[0]

            if best_tokens[0] == START_TOKEN:
                best_tokens = best_tokens[1:]
            if best_tokens[-1] == END_TOKEN:
                best_tokens = best_tokens[:-1]
            best_sentence = " ".join(best_tokens)

            print("\nbeam decoded sentence:", best_sentence)
            return best_sentence

    def inference_with_beamsearch_ranked_image(self, image, n_solutions=3):
        def compute_sim2image(current_text):
            # TODO: MELHORAR
            # condierar start and end_token?
            current_text = current_text[1:]  # ignore start token
            if current_text[-1] == END_TOKEN:
                current_text = current_text[:-1]
            n_tokens = len(current_text)

            tokens_ids = torch.zeros(1, n_tokens)
            for i in range(n_tokens):
                token = current_text[i]
                tokens_ids[0, i] = self.token_to_id[token]

            tokens_embeddings = self.decoder.embedding(tokens_ids.long()).to(self.device)
            sentence_mean = torch.mean(tokens_embeddings, dim=1)

            images_embedding = self.decoder.image_embedding

            return torch.cosine_similarity(sentence_mean, images_embedding)

        def generate_n_solutions(seed_text, seed_prob, encoder_out, h, c, n_solutions):
            last_token = seed_text[-1]

            if last_token == END_TOKEN:
                if len(seed_text) <= 2:
                    return [(seed_text, -np.inf, h, c)]
                return [(seed_text, seed_prob, h, c)]

            top_solutions = []
            scores, h, c = self.generate_output_index(
                torch.tensor([self.token_to_id[last_token]]), encoder_out, h, c)

            sorted_scores, sorted_indices = torch.sort(
                scores, descending=True, dim=-1)

            for index in range(n_solutions):
                text = seed_text + [self.id_to_token[sorted_indices[index].item()]]
                # beam search taking into account lenght of sentence
                # prob = (seed_prob*len(seed_text) + np.log(sorted_scores[index].item()) / (len(seed_text)+1))
                text_score = (seed_prob * len(seed_text) + np.log(sorted_scores[index].item())) / (len(seed_text) + 1)
                top_solutions.append((text, text_score, h, c))

            return top_solutions

        def get_most_probable(candidates, n_solutions):
            return sorted(candidates, key=operator.itemgetter(1), reverse=True)[:n_solutions]

        with torch.no_grad():
            # my_dict = {}

            encoder_output = self.encoder(image)
            encoder_output = encoder_output.view(1, -1, encoder_output.size()[-1])  # flatten encoder
            h, c = self.decoder.init_hidden_state(encoder_output)

            top_solutions = [([START_TOKEN], 0.0, h, c)]

            for time_step in range(self.max_len):
                candidates = []
                for sentence, prob, h, c in top_solutions:
                    candidates.extend(generate_n_solutions(
                        sentence, prob, encoder_output, h, c, n_solutions))

                top_solutions = get_most_probable(candidates, n_solutions)

                # print("\nall candidates", [(text, prob) for text, prob, _, _ in candidates])
                # # my_dict["cand"].append([(text, prob) for text, prob, _, _ in candidates])
                # print("\ntop", [(text, prob)
                #                 for text, prob, _, _ in top_solutions])
                # # my_dict["top"].append([(text, prob) for text, prob, _, _ in top_solutions])
                # my_dict[time_step] = {"cand": [(text, prob) for text, prob, _, _ in candidates],
                #                       "top": [(text, prob) for text, prob, _, _ in top_solutions]}

            # with open("beam_10.json", 'w+') as f:
            #     json.dump(my_dict, f, indent=2)
            # print("top solutions", [(text, prob)
            #                         for text, prob, _, _ in top_solutions])
            print("top before", [(text, prob)
                                 for text, prob, _, _ in top_solutions])
            final_solutions = []
            for sentence, prob, h, c in top_solutions:
                image_rank = compute_sim2image(sentence)
                final_solutions.append((sentence, image_rank))

            final_solutions = get_most_probable(final_solutions, n_solutions)
            print("final_solutions", final_solutions)

            best_tokens, prob = final_solutions[0]

            if best_tokens[0] == START_TOKEN:
                best_tokens = best_tokens[1:]
            if best_tokens[-1] == END_TOKEN:
                best_tokens = best_tokens[:-1]
            best_sentence = " ".join(best_tokens)

            print("\nbeam decoded sentence:", best_sentence)
            return best_sentence

    @abstractmethod
    def generate_output_index(self, input_word, encoder_out, h, c):
        pass
