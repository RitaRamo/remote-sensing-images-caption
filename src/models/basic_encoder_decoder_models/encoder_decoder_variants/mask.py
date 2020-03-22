import torchvision
from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from models.basic_encoder_decoder_models.encoder_decoder import BasicEncoderDecoderModel, Encoder
import torch.nn.functional as F
from embeddings.embeddings import get_embedding_layer
from models.abtract_model import AbstractEncoderDecoderModel
import random
from utils.early_stop import EarlyStopping
import numpy as np
import time
from preprocess_data.tokens import START_TOKEN, END_TOKEN
import logging


class Decoder(nn.Module):
    """
    Decoder.
    """

    def __init__(self, embedding_type, embed_dim, decoder_dim, vocab_size, token_to_id, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(Decoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        # self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.embedding = get_embedding_layer(
            embedding_type, embed_dim, vocab_size, token_to_id)

        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(
            embed_dim + embed_dim, decoder_dim, bias=True)  # decoding LSTMCell
        # linear layer to find initial hidden state of LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        # linear layer to find initial cell state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        #self.embedding.weight.data.uniform_(-0.1, 0.1)
        #print("e agora", self.embedding.weight.data)

        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def normalize_embeddings(self):
        """
        Normalize values of embbedings (ex: makes sense for pretrained embeddings)
        """

        embeddings_values = self.embedding.weight.data
        norm = embeddings_values.norm(
            p=2, dim=1, keepdim=True).clamp(min=1e-12)
        self.embedding.weight.data.copy_(embeddings_values.div(norm))

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, word_ground_truth, word_predicted, encoder_out, decoder_hidden_state, decoder_cell_state):
        # print("this is word_ground_truth", word_ground_truth)
        # print("this is word_predicted", word_predicted)

        embeddings_ground_truth = self.embedding(word_ground_truth)
        embeddings_predicted = self.embedding(word_predicted)

        decoder_input = torch.cat(
            (embeddings_ground_truth, embeddings_predicted), dim=1
        )

        decoder_hidden_state, decoder_cell_state = self.decode_step(
            decoder_input, (decoder_hidden_state, decoder_cell_state)
        )

        scores = self.fc(self.dropout(decoder_hidden_state))

        return scores, decoder_hidden_state, decoder_cell_state


class BasicMaskGroundTruthWithPredictionModel(BasicEncoderDecoderModel):
    MASK_TOKEN = "<mask>"

    def __init__(self,
                 args,
                 vocab_size,
                 token_to_id,
                 id_to_token,
                 max_len,
                 device
                 ):

        token_to_id[self.MASK_TOKEN] = vocab_size
        id_to_token[vocab_size] = self.MASK_TOKEN
        vocab_size = vocab_size+1

        super().__init__(args, vocab_size, token_to_id, id_to_token, max_len, device)

    def _initialize_encoder_and_decoder(self):

        self.encoder = Encoder(self.args.image_model_type,
                               enable_fine_tuning=self.args.fine_tune_encoder)

        self.decoder = Decoder(
            encoder_dim=self.encoder.encoder_dim,
            decoder_dim=self.args.decoder_dim,
            embedding_type=self.args.embedding_type,
            embed_dim=self.args.embed_dim,
            vocab_size=self.vocab_size,
            token_to_id=self.token_to_id,
            dropout=self.args.dropout
        )

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

    def train(self, train_dataloader, val_dataloader, print_freq):

        list_of_n_token_to_mask = [0, 2, 4, 8, 16, self.max_len-1]

        for n_tokens_to_mask in list_of_n_token_to_mask:

            early_stopping = EarlyStopping(
                epochs_limit_without_improvement=self.args.epochs_limit_without_improvement,
                epochs_since_last_improvement=0,
                baseline=np.Inf,
            )

            # Iterate by epoch
            for epoch in range(0, self.args.epochs):
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
                        imgs, caps, caplens, n_tokens_to_mask
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
                            imgs, caps, caplens, n_tokens_to_mask)

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

    def train_step(self, imgs, caps_input, cap_len, n_tokens_to_mask):
        encoder_out, caps_sorted, caption_lengths = self._prepare_inputs_to_forward_pass(
            imgs, caps_input, cap_len)

        predict_output = self._predict(
            encoder_out, caps_sorted, caption_lengths, n_tokens_to_mask)
        loss = self._calculate_loss(
            predict_output, caps_sorted, caption_lengths)

        self.decoder_optimizer.zero_grad()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        # if grad_clip is not None:
        #     clip_gradient(decoder_optimizer, grad_clip)
        #     if encoder_optimizer is not None:
        #         clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        self.decoder_optimizer.step()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.step()

        return loss

    def val_step(self, imgs, caps_input, cap_len, n_tokens_to_mask):
        encoder_out, caps_sorted, caption_lengths = self._prepare_inputs_to_forward_pass(
            imgs, caps_input, cap_len)

        predict_output = self._predict(
            encoder_out, caps_sorted, caption_lengths, n_tokens_to_mask)

        loss = self._calculate_loss(
            predict_output, caps_sorted, caption_lengths)

        return loss

    def _predict(self, encoder_out, caps, caption_lengths, n_tokens_to_mask):
        batch_size = encoder_out.size(0)

        # Create tensors to hold word predicion scores and alphas
        all_predictions = torch.zeros(batch_size, max(
            caption_lengths), self.vocab_size).to(self.device)

        h, c = self.decoder.init_hidden_state(encoder_out)

        # print("caps before requires grad",
        #       caps.requires_grad)

        caps = self._mask_caps(caps.clone().detach(), caption_lengths,
                               n_tokens_to_mask).to(self.device)

        decoder_predicted_word = torch.LongTensor(
            [self.token_to_id[START_TOKEN]] * batch_size).to(self.device)

        for t in range(max(
                caption_lengths)):
            # batchsizes of current time_step are the ones with lenght bigger than time-step (i.e have not fineshed yet)
            batch_size_t = sum([l > t for l in caption_lengths])

            predictions, h, c = self.decoder(
                caps[:batch_size_t, t], decoder_predicted_word[:batch_size_t], encoder_out[:batch_size_t], h[:batch_size_t], c[:batch_size_t])

            all_predictions[:batch_size_t, t, :] = predictions

            scores = F.log_softmax(predictions, dim=1)
            predicted_tokens = torch.argmax(scores, dim=1)
            decoder_predicted_word = predicted_tokens

        return {"predictions": all_predictions}

    def _calculate_loss(self, predict_output, caps_sorted, caption_lengths):
        torch.autograd.set_detect_anomaly(True)
        predictions = predict_output["predictions"]
        targets = caps_sorted[:, 1:]  # targets doesnt have stark token

        # pack scores and target
        predictions = pack_padded_sequence(
            predictions, caption_lengths, batch_first=True)
        targets = pack_padded_sequence(
            targets, caption_lengths, batch_first=True)

        loss = self.criterion(predictions.data, targets.data)

        return loss

    def _mask_caps(self, caps, caption_lengths, n_tokens_to_mask):
        mask = self.token_to_id[self.MASK_TOKEN]
        #print("caps before", caps)

        for batch_i in range(len(caps)):
            #print("this is batch i", batch_i)
            # random that does not consider start_token to mask, hence random(len(caps)-1) +1
            rand_columns = (torch.randperm(
                caption_lengths[batch_i]-1)+1)[:n_tokens_to_mask]
            caps[batch_i, rand_columns] = mask

        #print("caps after", caps)

        return caps

    def generate_text(self, image):
        with torch.no_grad():  # no need to track history

            decoder_sentence = START_TOKEN + " "

            input_word_id = torch.tensor([self.token_to_id[START_TOKEN]])
            predicted_word_id = torch.tensor([self.token_to_id[START_TOKEN]])

            i = 1

            encoder_output = self.encoder(image)
            encoder_output = encoder_output.view(
                1, -1, encoder_output.size()[-1])

            h, c = self.decoder.init_hidden_state(encoder_output)

            while True:

                predictions, h, c = self.decoder(
                    input_word_id, predicted_word_id, encoder_output, h, c)

                scores = F.log_softmax(predictions, dim=1)
                predicted_word_id = torch.argmax(scores, dim=1)

                predicted_word = self.id_to_token[predicted_word_id.item(
                )]

                decoder_sentence += " " + predicted_word

                if (predicted_word == END_TOKEN or
                        i >= self.max_len-1):  # until 35
                    break

                input_word_id = torch.tensor(
                    [self.token_to_id[self.MASK_TOKEN]])

                i += 1

            print("\ndecoded sentence", decoder_sentence)

            return decoder_sentence  # input_caption
