import operator
from data_preprocessing.preprocess_tokens import START_TOKEN, END_TOKEN, PAD_TOKEN
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
import torch.nn.functional as F


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

    def inference_with_greedy(self, image, n_solutions=0, min_len=0, repetition_window=0, max_len=50):
        scores_dict = {}
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

                # debug
                scores_dict[i] = {"sim": [], "sentence": []}
                scores_dict[i]["sim"] = ([(self.id_to_token[top_index.item()], scores.squeeze()[
                                         top_index.item()].item()) for top_index in sorted_indices.squeeze()[:10]])

                current_output_index = sorted_indices.squeeze()[0]

                current_output_token = self.id_to_token[current_output_index.item(
                )]

                decoder_sentence.append(current_output_token)

                scores_dict[i]["sentence"] = decoder_sentence

                if current_output_token == END_TOKEN:
                    # ignore end_token
                    decoder_sentence = decoder_sentence[:-1]
                    break

                if i >= self.max_len - 1:  # until 35
                    break

                input_word[0] = current_output_index.item()

                i += 1

            generated_sentence = " ".join(decoder_sentence)
            # print("beam_t decoded sentence:", generated_sentence)
            print("\ngenerated sentence:", generated_sentence)

            return generated_sentence, scores_dict  # input_caption

    def inference_with_greedy_debug(self, image, n_solutions=0, min_len=0, repetition_window=0, max_len=50):
        dict = {}
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

                current_output_index = sorted_indices.squeeze()[0]

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
            # print("beam_t decoded sentence:", generated_sentence)
            print("\ngenerated sentence:", generated_sentence)

            return generated_sentence  # input_caption

    def inference_with_greedy_smoothl1(self, image, n_solutions=0, min_len=0, repetition_window=0, max_len=50):
        with torch.no_grad():  # no need to track history

            decoder_sentence = []

            input_word = torch.tensor([self.token_to_id[START_TOKEN]])

            i = 1

            encoder_output = self.encoder(image)
            encoder_output = encoder_output.view(
                1, -1, encoder_output.size()[-1])

            h, c = self.decoder.init_hidden_state(encoder_output)

            criteria = torch.nn.SmoothL1Loss(reduction="none")

            while True:

                scores, h, c = self.generate_output_index_smoothl1(criteria,
                                                                   input_word, encoder_output, h, c)

                sorted_scores, sorted_indices = torch.sort(scores, descending=False, dim=-1)
                # print("this are the sorted_scores", sorted_scores)
                # print("this are the sorted_indices", sorted_indices)
                # k_l = 0
                # for indi in sorted_indices:
                #     print(self.id_to_token[indi.item()], sorted_scores[k_l])
                #     k_l += 1
                #     if k_l > 5:
                #         break

                current_output_index = sorted_indices.squeeze()[0]
                # print("current output index", current_output_index)
                # if current_output_index.item() == self.token_to_id[PAD_TOKEN]:
                #     current_output_index = sorted_indices.squeeze()[1]

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
            # print("beam_t decoded sentence:", generated_sentence)
            print("\ngenerated sentence:", generated_sentence)

            return generated_sentence  # input_caption

    def inference_with_beamsearch(self, image, n_solutions=3, min_len=2, repetition_window=0, max_len=50):

        def compute_probability(seed_text, seed_prob, sorted_scores, index, current_text):
            # print("seed text", seed_text)
            # print("current_text text", current_text)
            # print("previous seed prob", seed_prob)
            # print("now prob", sorted_scores[index].item())
            # print("final prob", seed_prob + sorted_scores[index].item())
            # return seed_prob + sorted_scores[index].item()
            return (seed_prob * len(seed_text) + sorted_scores[index].item()) / (len(seed_text) + 1)

        def generate_n_solutions(seed_text, seed_prob, encoder_out, h, c, n_solutions):
            last_token = seed_text[-1]

            if last_token == END_TOKEN:
                return [(seed_text, seed_prob, h, c)]

            if len(seed_text) > max_len:
                return [(seed_text, seed_prob, h, c)]

            top_solutions = []
            scores, h, c = self.generate_output_index(
                torch.tensor([self.token_to_id[last_token]]), encoder_out, h, c)
            sorted_scores, sorted_indices = torch.sort(
                scores.squeeze(), descending=True, dim=-1)

            # for index in range(n_solutions):
            #     text = seed_text + [self.id_to_token[sorted_indices[index].item()]]
            #     # beam search taking into account lenght of sentence
            #     # prob = (seed_prob*len(seed_text) + np.log(sorted_scores[index].item()) / (len(seed_text)+1))
            #     text_score = compute_probability(seed_text, seed_prob, sorted_scores, index, text)
            #     top_solutions.append((text, text_score, h, c))

            n = 0
            index = 0
            len_seed_text = len(seed_text)
            while n < n_solutions:
                current_word = self.id_to_token[sorted_indices[index].item()]
                if current_word == END_TOKEN:
                    if len(seed_text) <= min_len:
                        index += 1
                        continue
                elif current_word in seed_text[max(len_seed_text - repetition_window, 0):]:
                    index += 1
                    continue

                text = seed_text + [current_word]
                text_score = compute_probability(seed_text, seed_prob, sorted_scores, index, text)
                top_solutions.append((text, text_score, h, c))
                index += 1
                n += 1

            return top_solutions

        def get_most_probable(candidates, n_solutions):
            return sorted(candidates, key=operator.itemgetter(1), reverse=True)[:n_solutions]

        with torch.no_grad():
            # my_dict = {}

            encoder_output = self.encoder(image)
            encoder_output = encoder_output.view(1, -1, encoder_output.size()[-1])  # flatten encoder
            h, c = self.decoder.init_hidden_state(encoder_output)

            top_solutions = [([START_TOKEN], 0.0, h, c)]

            for time_step in range(self.max_len - 1):
                candidates = []
                for sentence, prob, h, c in top_solutions:
                    candidates.extend(generate_n_solutions(
                        sentence, prob, encoder_output, h, c, n_solutions))

                top_solutions = get_most_probable(candidates, n_solutions)

                # # print("\nall candidates", [(text, prob) for text, prob, _, _ in candidates])
                # my_dict["cand"].append([(text, prob) for text, prob, _, _ in candidates])
                # # print("\ntop", [(text, prob)
                # #                 for text, prob, _, _ in top_solutions])
                # my_dict["top"].append([(text, prob) for text, prob, _, _ in top_solutions])
                # my_dict[time_step] = {"cand": [(text, prob) for text, prob, _, _ in candidates],
                #                       "top": [(text, prob) for text, prob, _, _ in top_solutions]}

            # print("top solutions", [(text, prob)
            #                         for text, prob, _, _ in top_solutions])

            best_tokens, prob, h, c = top_solutions[0]

            # if np.isnan(prob):
            #     with open("beam_conMesmo.json", 'w+') as f:
            #         json.dump(my_dict, f, indent=2)
            #         print(stop)

            if best_tokens[0] == START_TOKEN:
                best_tokens = best_tokens[1:]
            if best_tokens[-1] == END_TOKEN:
                best_tokens = best_tokens[:-1]
            best_sentence = " ".join(best_tokens)

            print("\nbeam decoded sentence:", best_sentence)
            return best_sentence

    def inference_beam_without_refinement(
            self, image, n_solutions=3, min_len=2, repetition_window=0, max_len=50):

        def compute_probability(seed_text, seed_prob, sorted_scores, index, current_text):
            # print("\nseed text", seed_text)
            # print("current_text text", current_text)
            # print("previous seed prob", seed_prob)
            # print("now prob", sorted_scores[index].item())
            # print("final prob", seed_prob + sorted_scores[index])
            # print("final prob with item", seed_prob + sorted_scores[index].item())

            # print(stop)
            return seed_prob + sorted_scores[index]  # .item()

        def generate_n_solutions(seed_text, seed_prob, encoder_out, h, c, n_solutions):
            last_token = seed_text[-1]

            if last_token == END_TOKEN:
                return [(seed_text, seed_prob, h, c)]

            if len(seed_text) > max_len:
                return [(seed_text, seed_prob, h, c)]

            top_solutions = []
            scores, h, c = self.generate_output_index(
                torch.tensor([self.token_to_id[last_token]]), encoder_out, h, c)

            sorted_scores, sorted_indices = torch.sort(
                scores.squeeze(), descending=True, dim=-1)

            #sorted_scores, sorted_indices = scores.squeeze().topk(self.vocab_size, 0, True, True)
            # print("sorted scores", sorted_scores)
            # print("sorted_indices", sorted_indices)

            # top_k_scores, top_k_words = scores.squeeze().topk(n_solutions, 0, True, True)  # (s)

            # # print("top ksie", top_k_scores.size())
            # # print("scores size", scores.size())

            # top_k_zero = torch.zeros(scores.squeeze().size()[0]).to(self.device)
            # print("top k zero", top_k_zero)
            # print("setp 1 top_k_scores", top_k_scores)
            # print("setp 1 top_k_words", top_k_words)
            # print("ste1 tok j score without item", 0.0 + top_k_scores[0])
            # print("ste1 tok j score item", 0.0 + top_k_scores[0].item())
            # print("ste1 tok j score item without 0 ", top_k_scores[0].item())
            # #print("ste1 top_k_zero score item", (top_k_zero + top_k_scores)[0].item())
            # print("ste1 top_k_zero score item", (top_k_zero + scores.squeeze()).topk(n_solutions, 0, True, True))

            # print("wit it", (top_k_zero + scores.squeeze()).topk(n_solutions, 0, True, True)[0][0].item())

            # print("sorted scores 0", sorted_scores[0])
            # print("sorted_indices 0", sorted_indices[0])

            # top_k_scores, top_k_words = scores.squeeze().topk(n_solutions, 0, True, True)  # (s)
            # print("setp 1 top_k_scores 0", top_k_scores[0])
            # print("setp 1 top_k_words 0", top_k_words[0])
            # print(stop)

            # for index in range(n_solutions):
            #     text = seed_text + [self.id_to_token[sorted_indices[index].item()]]
            #     # beam search taking into account lenght of sentence
            #     # prob = (seed_prob*len(seed_text) + np.log(sorted_scores[index].item()) / (len(seed_text)+1))
            #     text_score = compute_probability(seed_text, seed_prob, sorted_scores, index, text)
            #     top_solutions.append((text, text_score, h, c))

            n = 0
            index = 0
            len_seed_text = len(seed_text)
            # print("\n start candidates")
            while n < n_solutions:
                current_word = self.id_to_token[sorted_indices[index].item()]
                if current_word == END_TOKEN:
                    if len(seed_text) <= min_len:
                        index += 1
                        continue
                elif current_word in seed_text[max(len_seed_text - repetition_window, 0):]:
                    index += 1
                    continue

                text = seed_text + [current_word]
                text_score = compute_probability(seed_text, seed_prob, sorted_scores, index, text)
                top_solutions.append((text, text_score, h, c))
                index += 1
                n += 1

            return top_solutions

        def get_most_probable(candidates, n_solutions):
            return sorted(candidates, key=operator.itemgetter(1), reverse=True)[:n_solutions]

        with torch.no_grad():
            my_dict = {}

            encoder_output = self.encoder(image.to(self.device))
            encoder_output = encoder_output.view(1, -1, encoder_output.size()[-1])  # flatten encoder
            h, c = self.decoder.init_hidden_state(encoder_output)

            top_solutions = [([START_TOKEN], 0.0, h, c)]

            for time_step in range(self.max_len - 1):
                # print("\nnew time step")
                candidates = []
                for sentence, prob, h, c in top_solutions:
                    candidates.extend(generate_n_solutions(
                        sentence, prob, encoder_output, h, c, n_solutions))

                top_solutions = get_most_probable(candidates, n_solutions)

                # print("\ntop", [(text, prob) for text, prob, _, _ in top_solutions])

                # # print("\nall candidates", [(text, prob) for text, prob, _, _ in candidates])
                # my_dict["cand"].append([(text, prob) for text, prob, _, _ in candidates])
                # # print("\ntop", [(text, prob)
                # #                 for text, prob, _, _ in top_solutions])
                # my_dict["top"].append([(text, prob) for text, prob, _, _ in top_solutions])
                # my_dict[time_step] = {"cand": [(text, prob.item()) for text, prob, _, _ in candidates],
                #                       "top": [(text, prob.item()) for text, prob, _, _ in top_solutions]}

            # print("top solutions", [(text, prob)
            #                         for text, prob, _, _ in top_solutions])

            best_tokens, prob, h, c = top_solutions[0]

            # if np.isnan(prob):
            #     with open("beam_conMesmo.json", 'w+') as f:
            #         json.dump(my_dict, f, indent=2)
            #         print(stop)

            if best_tokens[0] == START_TOKEN:
                best_tokens = best_tokens[1:]
            if best_tokens[-1] == END_TOKEN:
                best_tokens = best_tokens[:-1]
            best_sentence = " ".join(best_tokens)

            print("\nbeam decoded sentence:", best_sentence)
            return best_sentence

    def inference_beam_tutorial(self, image, n_solutions=3, min_len=2, repetition_window=0, max_len=50):
        my_dict = {}

        k = n_solutions

        # Move to GPU device, if available
        image = image.to(self.device)  # (1, 3, 256, 256)

        # Encode
        encoder_out = self.encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)
        # print("encoder out", encoder_out.size())
        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
        # print("encoder out", encoder_out.size())

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[self.token_to_id[START_TOKEN]]] * k).to(self.device)  # (k, 1)
        # print("k_prev_words", k_prev_words)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)
        # print("seqs", seqs)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(self.device)  # (k, 1)
        # print("top_k_scores", top_k_scores)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = self.decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            # embeddings = self.decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            # awe, _ = self.decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
            # print("awe size", awe.size())
            # # gate = self.decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            # #awe = gate * awe
            # h, c = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
            # scores = self.decoder.fc(h)  # (s, vocab_size)
            # print("j_prev words", k_prev_words)
            # print("prev word size", k_prev_words.squeeze(1).size())

            scores, h, c = self.generate_output_index(k_prev_words.squeeze(1), encoder_out, h, c)

            # scores = F.log_softmax(scores, dim=1)
            # Add
            # print("\nscores do log", scores.size())
            # print("scores do log", scores)
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
            # print("expand top k scores", top_k_scores.expand_as(scores))
            # print("final scores do top k (sum expand com scores log)", scores)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                # print("setp 1 top_k_scores", top_k_scores)
                # print("setp 1 top_k_words", top_k_words)
                # print("setp 1 top_k_scores item", top_k_scores[0].item())

                # print(stop)

            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)
                # print("scores view", scores.view(-1))
                # print("setp 2 top_k_scores", top_k_scores)
                # print("setp 2 top_k_words", top_k_words)

            # your code TODO:REMOVE
            cands = []
            id_score = 0
            for score in scores:
                # PRINT ISTO SÃO OS CANIDATOS COMO TENS NO TEU!!
                fake_top_k_scores, fake_top_k_words = score.topk(k, 0, True, True)  # (s)
                # print("fake top_k_scores", fake_top_k_scores)
                # print("fake top_k_words", fake_top_k_words)
                # my_dict[time_step] = {"cand": [(text, prob) for text, prob, _, _ in candidates],
                # "top": [(text, prob) for text, prob, _, _ in top_solutions]}

                for n_possibles in range(k):
                    #print("n poss", n_possibles)
                    cand_sentence = [self.id_to_token[index.item()] for index in seqs[id_score]
                                     ] + [self.id_to_token[fake_top_k_words[n_possibles].item()]]
                    cand_sentence_score = fake_top_k_scores[n_possibles].item()
                    cands.append((cand_sentence, cand_sentence_score))
                    #print("this is cand sente", (cand_sentence, cand_sentence_score))

                #print("all cands", cands)

                # "top": [(text, prob) for text, prob, _, _ in top_solutions]
                id_score += 1
                # print(stop)

            # Convert unrolled indices to actual indices of scores
            # print("top k word", top_k_words)
            prev_word_inds = top_k_words / self.vocab_size  # (s)
            # print("what is prev word ind", prev_word_inds)
            next_word_inds = top_k_words % self.vocab_size  # (s)
            # print("what is next_word_inds", next_word_inds)

            #print("seqs[prev_word_inds]", seqs[prev_word_inds])
            #print("word ind uns", next_word_inds.unsqueeze(1))

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            #print("cat seqs", seqs)

            # MINE
            my_top_of_cand = []
            id_score = 0
            for seq in seqs:
                #print("printed seq", [self.id_to_token[index.item()] for index in seq])

                # YOUR CODE:TODO REMOVE
                my_top_of_cand.append(([self.id_to_token[index.item()]
                                        for index in seq], top_k_scores[id_score].item()))
                id_score += 1

            # TODO:REMOVE YOUR CODE
            # if step > 1:

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != self.token_to_id[END_TOKEN]]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
            #print("incomplete_inds", incomplete_inds)
            #print("completed", complete_inds)

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])

                #TODO: remover
                for c_i in complete_inds:
                    my_top_of_cand.append(([self.id_to_token[index.item()]
                                            for index in seqs[c_i]], top_k_scores[c_i].item()))
            k -= len(complete_inds)  # reduce beam length accordingly

            # my_dict[step] = {
            #     "cand": cands,
            #     "top": my_top_of_cand
            # }
            # print("\n FINAL dict", my_dict)
            # Proceed with incomplete sequences
            if k == 0:
                # print("entrei aqui")
                break
            seqs = seqs[incomplete_inds]
            # print("prev_word_inds[incomplete_inds]", prev_word_inds[incomplete_inds])
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            #print("top k scores before transformed", top_k_scores)
            #print("top k scores will be transformer", top_k_scores[incomplete_inds].unsqueeze(1))

            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            #print("k_prev_words", next_word_inds[incomplete_inds].unsqueeze(1))

            # Break if things have been going on too long
            if step >= self.max_len - 1:
                # print("I reached max len")
                # complete_seqs = seqs
                break

            step += 1
        # print("compete seq", complete_seqs_scores)
        # If MAX CAPTION LENGTH has been reached and no sequence has reached the eoc
        # there will be no complete seqs, use the incomplete ones
        if k == n_solutions:
            complete_seqs.extend(seqs[[incomplete_inds]].tolist())
            complete_seqs_scores.extend(top_k_scores[[incomplete_inds]])

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq_index = complete_seqs[i]
        best_tokens = [self.id_to_token[index] for index in seq_index]

        #print("completed seq", complete_seqs)
        # for seq in complete_seqs:
        #print("final comleted", [self.id_to_token[index] for index in seq])
        #print("comple complete_seqs_scores", complete_seqs_scores)
        #print("i completed seq score", i)

        if best_tokens[0] == START_TOKEN:
            best_tokens = best_tokens[1:]
        if best_tokens[-1] == END_TOKEN:
            best_tokens = best_tokens[: -1]
        best_sentence = " ".join(best_tokens)

        print("beam_t decoded sentence:", best_sentence)

        return best_sentence

    def inference_beam_comp(
        self,
        image,
        n_solutions=3,
        min_len=2,
        repetition_window=0,
        max_len=50,
        store_alphas=False,
        store_beam=False,
        print_beam=False,
    ):
        """Generate and return the top k sequences using beam search."""
        my_dict = {}

        beam_size = n_solutions
        current_beam_width = n_solutions
        encoder_output = self.encoder(image)
        enc_image_size = encoder_output.size(1)
        encoder_dim = encoder_output.size()[-1]

        # Flatten encoding
        encoder_output = encoder_output.view(1, -1, encoder_dim)

        # We'll treat the problem as having a batch size of k
        encoder_output = encoder_output.expand(
            beam_size, encoder_output.size(1), encoder_dim
        )

        # Tensor to store top k sequences; now they're just <start>
        top_k_sequences = torch.full(
            (beam_size, 1), self.token_to_id[START_TOKEN], dtype=torch.int64, device=self.device
        )

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(beam_size, device=self.device)

        if store_alphas:
            # Tensor to store top k sequences' alphas; now they're just 1s
            seqs_alpha = torch.ones(beam_size, 1, enc_image_size, enc_image_size).to(
                self.device
            )

        # Lists to store completed sequences, scores, and alphas and the full decoding beam
        complete_seqs = []
        complete_seqs_alpha = []
        complete_seqs_scores = []
        beam = []

        # Initialize hidden states
        # states = self.decoder.init_hidden_states(encoder_output)
        decoder_hidden_state, decoder_cell_state = self.decoder.init_hidden_state(encoder_output)
        states = [decoder_hidden_state, decoder_cell_state]
        # Start decoding
        for step in range(0, self.max_len - 1):
            prev_words = top_k_sequences[:, step]
            # print("prev word size", prev_words.size())
            # print("\nvou começar para estas palavras", prev_words)
            # prev_word_embeddings = self.word_embedding(prev_words)

            # word, encoder_out, decoder_hidden_state, decoder_cell_state
            # scores, decoder_hidden_state, decoder_cell_state, alpha = self.decoder.forward(
            #     prev_words, encoder_output, states[0], states[1]
            # )
            scores, decoder_hidden_state, decoder_cell_state = self.generate_output_index(
                prev_words, encoder_output, states[0], states[1])
            states = [decoder_hidden_state, decoder_cell_state]
            # scores = F.log_softmax(predictions, dim=1)

            # Add the new scores
            scores = top_k_scores.unsqueeze(1).expand_as(scores) + scores

            # For the first timestep, the scores from previous decoding are all the same, so in order to create 5
            # different sequences, we should only look at one branch
            if step == 0:
                scores = scores[0]

            # if step > 2:
            #     stop

            # Find the top k of the flattened scores
            top_k_scores, top_k_words = scores.view(-1).topk(
                current_beam_width, 0, largest=True, sorted=True
            )

            # # YOUR CODE FOR DEBUGGING
            # if step > 0:
            #     # MY DEBUG
            #     cands = []
            #     # top=
            #     id_score = 0
            #     for score in scores:
            #         # PRINT ISTO SÃO OS CANIDATOS COMO TENS NO TEU!!
            #         fake_top_k_scores, fake_top_k_words = score.topk(current_beam_width, 0, True, True)  # (s)
            #         print("fake top_k_scores", fake_top_k_scores)
            #         print("fake top_k_words", fake_top_k_words)
            #         # my_dict[time_step] = {"cand": [(text, prob) for text, prob, _, _ in candidates],
            #         # "top": [(text, prob) for text, prob, _, _ in top_solutions]}

            #         for n_possibles in range(current_beam_width):
            #             print("n poss", n_possibles)
            #             cand_sentence = [self.id_to_token[index.item()] for index in top_k_sequences[id_score]
            #                              ] + [self.id_to_token[fake_top_k_words[n_possibles].item()]]
            #             cand_sentence_score = fake_top_k_scores[n_possibles].item()
            #             cands.append((cand_sentence, cand_sentence_score))
            #             print("this is cand sente", (cand_sentence, cand_sentence_score))

            #         print("all cands", cands)

            #         # "top": [(text, prob) for text, prob, _, _ in top_solutions]
            #         id_score += 1

            # print("top_k_scores", top_k_scores)
            # print("top_k_words", top_k_words)

            # Convert flattened indices to actual indices of scores
            prev_seq_inds = top_k_words / self.vocab_size  # (k)
            next_words = top_k_words % self.vocab_size  # (k)

            # print("prev_seq_inds", prev_seq_inds)
            # print("next_words", next_words)

            # print("prev ", prev_seq_inds)
            # print("next_words ", next_words)

            # Add new words to sequences
            top_k_sequences = torch.cat(
                (top_k_sequences[prev_seq_inds], next_words.unsqueeze(1)), dim=1
            )
            # print("top k senten", top_k_sequences)

            # print("top k sentences", top_k_sequences)

            # if print_beam:
            #     print_current_beam(top_k_sequences, top_k_scores, self.word_map)
            if store_beam:
                beam.append(top_k_sequences)

            # print("\nmy beam top")
            # for sequence, score in zip(top_k_sequences, top_k_scores):
            #     print("sequence", sequence)
            #     print("printed seq", [self.id_to_token[index.item()] for index in sequence], score)
            #     print(sequence, score)

            id_score = 0
            my_top_of_cand = []
            for seq in top_k_sequences:
                #print("printed seq", [self.id_to_token[index.item()] for index in seq])

                my_top_of_cand.append(([self.id_to_token[index.item()]
                                        for index in seq], top_k_scores[id_score].item()))
                id_score += 1

            # if step > 0:
            #     my_dict[step] = {
            #         "cand": cands,
            #         "top": my_top_of_cand
            #     }
            #     print("\n FINAL dict", my_dict)

                # Store the new alphas
            if store_alphas:
                alpha = alpha.view(-1, enc_image_size, enc_image_size)
                seqs_alpha = torch.cat(
                    (seqs_alpha[prev_seq_inds], alpha[prev_seq_inds].unsqueeze(1)),
                    dim=1,
                )

            # Check for complete and incomplete sequences (based on the <end> token)
            incomplete_inds = (
                torch.nonzero(next_words != self.token_to_id[END_TOKEN]).view(-1).tolist()
            )
            # print("incomple ind", incomplete_inds)

            complete_inds = (
                torch.nonzero(next_words == self.token_to_id[END_TOKEN]).view(-1).tolist()
            )
            # print("complete_inds", complete_inds)

            # Set aside complete sequences and reduce beam size accordingly
            if len(complete_inds) > 0:
                complete_seqs.extend(top_k_sequences[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
                if store_alphas:
                    complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())

            # Stop if k captions have been completely generated
            current_beam_width = len(incomplete_inds)
            if current_beam_width == 0:
                break

            # Proceed with incomplete sequences
            top_k_sequences = top_k_sequences[incomplete_inds]
            # print("top k sentences", top_k_sequences)
            for i in range(len(states)):
                states[i] = states[i][prev_seq_inds[incomplete_inds]]
            encoder_output = encoder_output[prev_seq_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds]
            if store_alphas:
                seqs_alpha = seqs_alpha[incomplete_inds]

        if len(complete_seqs) < beam_size:
            complete_seqs.extend(top_k_sequences.tolist())
            complete_seqs_scores.extend(top_k_scores)
            if store_alphas:
                complete_seqs_alpha.extend(seqs_alpha)

        sorted_sequences = [
            sequence
            for _, sequence in sorted(
                zip(complete_seqs_scores, complete_seqs), reverse=True
            )
        ]
        sorted_alphas = None
        if store_alphas:
            sorted_alphas = [
                alpha
                for _, alpha in sorted(
                    zip(complete_seqs_scores, complete_seqs_alpha), reverse=True
                )
            ]
        # print("sorted seq", sorted_sequences)

        best_tokens = [self.id_to_token[index] for index in sorted_sequences[0]]

        if best_tokens[0] == START_TOKEN:
            best_tokens = best_tokens[1:]
        if best_tokens[-1] == END_TOKEN:
            best_tokens = best_tokens[:-1]
        best_sentence = " ".join(best_tokens)

        print("beam_t decoded sentence:", best_sentence)

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
            my_dict = {}

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

                print("\nall candidates", [(text, prob) for text, prob, _, _ in candidates])
                # my_dict["cand"].append([(text, prob) for text, prob, _, _ in candidates])
                print("\ntop", [(text, prob)
                                for text, prob, _, _ in top_solutions])
                # my_dict["top"].append([(text, prob) for text, prob, _, _ in top_solutions])
                my_dict[time_step] = {"cand": [(text, prob) for text, prob, _, _ in candidates],
                                      "top": [(text, prob) for text, prob, _, _ in top_solutions]}

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
