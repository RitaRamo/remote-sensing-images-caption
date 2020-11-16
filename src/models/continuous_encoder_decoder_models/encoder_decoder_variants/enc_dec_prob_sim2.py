import torchvision
from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from models.basic_encoder_decoder_models.encoder_decoder import Encoder, Decoder
from models.abtract_model import AbstractEncoderDecoderModel
import torch.nn.functional as F
from embeddings.embeddings import get_embedding_layer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from data_preprocessing.preprocess_tokens import OOV_TOKEN
from embeddings.embeddings import EmbeddingsType
from models.continuous_encoder_decoder_models.encoder_decoder import ContinuousEncoderDecoderModel
from embeddings.embeddings import EmbeddingsType
from utils.optimizer import get_optimizer, clip_gradient


class ContinuousDecoder(Decoder):

    def __init__(self, decoder_dim, embed_dim, embedding_type, vocab_size, token_to_id, post_processing,
                 encoder_dim=2048, dropout=0.5, device=None):

        super(ContinuousDecoder, self).__init__(decoder_dim, embed_dim, embedding_type,
                                                vocab_size, token_to_id, post_processing, encoder_dim, dropout)

        # linear layer to find representation of image
        # replace softmax with a embedding layer
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.softmax = nn.Softmax(dim=1)

        list_wordid = list(range(vocab_size))  # ignore first 4 special tokens : "start,end, unknow, padding"

        vocab = torch.transpose(torch.tensor(list_wordid).unsqueeze(-1), 0, 1)
        self.embedding_vocab = self.embedding(vocab).to(device)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)

        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)

        return h, h

    def forward(self, word, encoder_out, decoder_hidden_state, decoder_cell_state):
        embeddings = self.embedding(word)

        decoder_hidden_state, decoder_cell_state = self.decode_step(
            embeddings, (decoder_hidden_state, decoder_cell_state)
        )

        scores = self.fc(self.dropout(decoder_hidden_state))

        alpha = self.softmax(scores)  # (batch_size, l_regions)
        vocab = self.embedding_vocab.repeat(decoder_hidden_state.size()[0], 1, 1)
        attention_weighted_encoding = (
            vocab * alpha.unsqueeze(2)).sum(dim=1)

        return scores, attention_weighted_encoding, decoder_hidden_state, decoder_cell_state


class ContinuousEncoderDecoderProbSim2Model(ContinuousEncoderDecoderModel):

    def __init__(self,
                 args,
                 vocab_size,
                 token_to_id,
                 id_to_token,
                 max_len,
                 device
                 ):
        super().__init__(args, vocab_size, token_to_id, id_to_token, max_len, device)

    def _initialize_encoder_and_decoder(self):

        if (self.args.embedding_type not in [embedding.value for embedding in EmbeddingsType]):
            raise ValueError(
                "Continuous model should use pretrained embeddings...")

        self.encoder = Encoder(self.args.image_model_type,
                               enable_fine_tuning=self.args.fine_tune_encoder)

        self.decoder = ContinuousDecoder(
            encoder_dim=self.encoder.encoder_dim,
            decoder_dim=self.args.decoder_dim,
            embedding_type=self.args.embedding_type,
            embed_dim=self.args.embed_dim,
            vocab_size=self.vocab_size,
            token_to_id=self.token_to_id,
            post_processing=self.args.post_processing,
            dropout=self.args.dropout,
            device=self.device
        )

        self.decoder.normalize_embeddings(self.args.no_normalization)

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        if self.args.grad_norm:
            self.loss_weight_ce = torch.ones(
                1, requires_grad=True, device=self.device, dtype=torch.float
            )
            self.loss_weight_sim = torch.ones(
                1, requires_grad=True, device=self.device, dtype=torch.float
            )

            self.loss_weight_sent = torch.ones(
                1, requires_grad=True, device=self.device, dtype=torch.float
            )

            self.gradnorm_optimizer = torch.optim.Adam(
                [self.loss_weight_ce, self.loss_weight_sim, self.loss_weight_sent],
                lr=0.025,
            )
            self.gradnorm_loss = nn.L1Loss().to(self.device)
        else:
            self.loss_weight_ce = torch.ones(
                1, device=self.device, dtype=torch.float
            )
            self.loss_weight_sim = torch.ones(
                1, device=self.device, dtype=torch.float
            )
            self.loss_weight_sent = torch.ones(
                1, device=self.device, dtype=torch.float
            )
        self.initial = False
        print("AQUI loss weights ce", self.loss_weight_ce.item())
        print("AWUI loss weights sim", self.loss_weight_sim.item())

    def _define_loss_criteria(self):
        self.criterion_ce = nn.CrossEntropyLoss().to(self.device)
        self.criterion_sim = nn.CosineEmbeddingLoss().to(self.device)

    def _predict(self, encoder_out, caps, caption_lengths):
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)

        # Create tensors to hold word predicion scores and alphas
        all_predictions1 = torch.zeros(batch_size, max(
            caption_lengths), self.vocab_size).to(self.device)
        all_predictions2 = torch.zeros(batch_size, max(
            caption_lengths), self.decoder.embed_dim).to(self.device)

        h, c = self.decoder.init_hidden_state(encoder_out)

        # Predict
        for t in range(max(
                caption_lengths)):
            # batchsizes of current time_step are the ones with lenght bigger than time-step (i.e have not fineshed yet)
            batch_size_t = sum([l > t for l in caption_lengths])

            predictions1, predictions2, h, c = self.decoder(
                caps[:batch_size_t, t], encoder_out[:batch_size_t], h[:batch_size_t], c[:batch_size_t])

            all_predictions1[:batch_size_t, t, :] = predictions1
            all_predictions2[:batch_size_t, t, :] = predictions2

        return {"predictions1": all_predictions1, "predictions2": all_predictions2}

    def _calculate_loss(self, predict_output, caps, caption_lengths):
        predictions = predict_output["predictions1"]
        targets = caps[:, 1:]  # targets doesnt have stark token
        target_embeddings = self.decoder.embedding(targets).to(self.device)

        predictions_embeddings = predict_output["predictions2"]

        if self.args.no_normalization == False:
            # when target embeddings start normalized, predictions should also be normalized
            predictions_embeddings = torch.nn.functional.normalize(predictions_embeddings, p=2, dim=-1)

        word_losses = 0.0  # pred_against_target_loss; #pred_sentence_again_target_sentence;"pred_sentence_agains_image
        word_ce_losses = 0.0
        sentence_losses = 0.0

        n_sentences = predictions_embeddings.size()[0]
        for i in range(n_sentences):  # iterate by sentence
            preds1_without_padd = predictions[i, :caption_lengths[i]]
            targets1_without_padd = targets[i, :caption_lengths[i]]
            word_ce_losses += self.criterion_ce(preds1_without_padd, targets1_without_padd)

            preds_without_padd = predictions_embeddings[i, :caption_lengths[i], :]
            targets_without_padd = target_embeddings[i, :caption_lengths[i], :]

            y = torch.ones(targets_without_padd.shape[0]).to(self.device)

            # word-level loss   (each prediction against each target)
            word_losses += self.criterion_sim(
                preds_without_padd,
                targets_without_padd,
                y
            )

            # sentence-level loss (sentence predicted agains target sentence)
            sentence_mean_pred = torch.mean(preds_without_padd, dim=0).unsqueeze(0)  # ver a dim
            sentece_mean_target = torch.mean(targets_without_padd, dim=0).unsqueeze(0)

            y = torch.ones(1).to(self.device)

            sentence_losses += self.criterion_sim(
                sentence_mean_pred,
                sentece_mean_target,
                y
            )

        word_ce_loss = word_ce_losses / n_sentences
        word_loss = word_losses / n_sentences
        sentence_loss = sentence_losses / n_sentences

        if self.initial == False:
            self.initial = True
            self.initial_ce_loss = word_ce_loss
            self.initial_sim_loss = word_loss
            self.initial_sent_loss = sentence_loss

        #loss = word_ce_loss + self.args.w2 * word_loss + sentence_loss

        return word_ce_loss, word_loss, sentence_loss

    def val_step(self, imgs, caps_input, cap_len, all_captions):
        (loss_ce, loss_sim, loss_sent), hypotheses, references_without_padding = super().val_step(
            imgs, caps_input, cap_len, all_captions)
        loss = self.loss_weight_ce[0].data * loss_ce + self.loss_weight_sim[0].data * \
            loss_sim + self.loss_weight_sent[0].data * loss_sent
        print("weight ce", self.loss_weight_ce[0].data)
        print("weight sim", self.loss_weight_sim[0].data)
        print("weight sent", self.loss_weight_sent[0].data)
        return loss, hypotheses, references_without_padding

    def train_step(self, imgs, caps_input, cap_len):
        encoder_out, caps_sorted, caption_lengths, sort_ind = self._prepare_inputs_to_forward_pass(
            imgs, caps_input, cap_len)

        predict_output = self._predict(
            encoder_out, caps_sorted, caption_lengths)

        loss_ce, loss_sim, loss_sent = self._calculate_loss(
            predict_output, caps_sorted, caption_lengths)

        loss = self.loss_weight_ce[0] * loss_ce + self.loss_weight_sim[0] * loss_sim + self.loss_weight_sent[0] * loss_sent

        self.decoder_optimizer.zero_grad()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()

        loss.backward(retain_graph=self.args.grad_norm)

        if self.args.grad_norm:
            self.apply_grad_norm(loss_ce, loss_sim, loss_sent)

        # # Clip gradients
        clip_gradient(self.decoder_optimizer, 5.)
        if self.encoder_optimizer is not None:
            clip_gradient(self.encoder_optimizer, 5.)

        # Update weights
        self.decoder_optimizer.step()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.step()

        return loss

    def apply_grad_norm(self, loss_ce, loss_sim, loss_sent):

        G1R = torch.autograd.grad(
            loss_ce, self.decoder.fc.parameters(), retain_graph=True, create_graph=True
        )

        G1R_flattened = torch.cat([g.view(-1) for g in G1R])
        G1 = torch.norm(self.loss_weight_ce * G1R_flattened.detach(), 2).unsqueeze(0)

        G2R = torch.autograd.grad(loss_sim, self.decoder.fc.parameters(), retain_graph=True)
        G2R_flattened = torch.cat([g.view(-1) for g in G2R])
        G2 = torch.norm(self.loss_weight_sim * G2R_flattened.detach(), 2).unsqueeze(0)

        G3R = torch.autograd.grad(loss_sent, self.decoder.fc.parameters())
        G3R_flattened = torch.cat([g.view(-1) for g in G3R])
        G3 = torch.norm(self.loss_weight_sent * G3R_flattened.detach(), 2).unsqueeze(0)

        # Calculate the average gradient norm across all tasks
        G_avg = torch.div(G1 + G2 + G3, 3)

        # Calculate relative losses
        lhat1 = torch.div(loss_ce.detach(), self.initial_ce_loss)
        lhat2 = torch.div(loss_sim.detach(), self.initial_sim_loss)
        lhat3 = torch.div(loss_sent.detach(), self.initial_sent_loss)
        lhat_avg = torch.div(lhat1 + lhat2 + lhat3, 3)

        # Calculate relative inverse training rates
        inv_rate1 = torch.div(lhat1, lhat_avg)
        inv_rate2 = torch.div(lhat2, lhat_avg)
        inv_rate3 = torch.div(lhat3, lhat_avg)

        # Calculate the gradient norm target for this batch
        C1 = G_avg * (inv_rate1 ** self.args.grad_norm_alpha)
        C2 = G_avg * (inv_rate2 ** self.args.grad_norm_alpha)
        C3 = G_avg * (inv_rate3 ** self.args.grad_norm_alpha)

        C1 = C1.detach()
        C2 = C2.detach()
        C3 = C3.detach()

        # Backprop and perform an optimization step
        self.gradnorm_optimizer.zero_grad()
        # Calculate the gradnorm loss
        Lgrad = self.gradnorm_loss(G1, C1) + self.gradnorm_loss(G2, C2) + self.gradnorm_loss(G3, C3)
        Lgrad.backward()
        self.gradnorm_optimizer.step()

        coef = 3 / (self.loss_weight_ce + self.loss_weight_sim + self.loss_weight_sent)
        self.loss_weight_ce.data = coef.data * self.loss_weight_ce.data
        self.loss_weight_sim.data = coef.data * self.loss_weight_sim.data
        self.loss_weight_sent.data = coef.data * self.loss_weight_sent.data

    def generate_output_index(self, input_word, encoder_out, h, c):
        predictions1, predictions2, h, c = self.decoder(
            input_word, encoder_out, h, c)

        current_output_index = self._convert_prediction_to_output(predictions1)

        return current_output_index, h, c

    def _convert_prediction_to_output(self, predictions):
        scores = F.log_softmax(predictions, dim=1)  # more stable
        # scores = F.softmax(predictions, dim=1)[0]  # actually probs
        return scores

    # def generate_output_index(self, input_word, encoder_out, h, c):
    #     predictions1, predictions2, h, c = self.decoder(
    #         input_word, encoder_out, h, c)

    #     current_output_index = self._convert_prediction_to_output(predictions2)

    #     return current_output_index, h, c

    # def _convert_prediction_to_output(self, predictions):
    #     output = torch.cosine_similarity(
    #         self.decoder.embedding.weight.data, predictions.unsqueeze(1), dim=-1)
    #     # scores = F.softmax(predictions, dim=1)[0]  # actually probs
    #     return output
