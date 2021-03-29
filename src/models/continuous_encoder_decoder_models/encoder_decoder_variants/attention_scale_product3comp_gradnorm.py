import torchvision
from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from models.basic_encoder_decoder_models.encoder_decoder_variants.attention_scale_product import ScaleProductAttention, Encoder, DecoderWithAttention
from models.abtract_model import AbstractEncoderDecoderModel
import torch.nn.functional as F
from embeddings.embeddings import get_embedding_layer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from data_preprocessing.preprocess_tokens import OOV_TOKEN, START_TOKEN, END_TOKEN
from embeddings.embeddings import EmbeddingsType
from models.continuous_encoder_decoder_models.encoder_decoder import ContinuousEncoderDecoderModel
from embeddings.embeddings import EmbeddingsType
from data_preprocessing.preprocess_images import get_image_model
from utils.enums import ContinuousLossesType
from utils.optimizer import get_optimizer, clip_gradient
from utils.early_stop import EarlyStopping
import time
import logging
from definitions_datasets import PATH_TRAINED_MODELS, PATH_EVALUATION_SCORES
import json
import operator 

class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, model_type, encoded_image_size=14, enable_fine_tuning=False):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        self.model_type = model_type

        self.model, self.encoder_dim = get_image_model(model_type)
        # resnet = torchvision.models.resnet101(
        #     pretrained=True)  # pretrained ImageNet ResNet-101

        # # Remove linear and pool layers (since we're not doing classification)
        # modules = list(resnet.children())[:-2]
        # self.model = nn.Sequential(*modules)
        # self.encoder_dim = 2048

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size))

        # Disable calculation of all gradients
        for p in self.model.parameters():
            p.requires_grad = False

        # Enable calculation of some gradients for fine tuning
        self.fine_tune(enable_fine_tuning)

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        # out = self.model(
        #     images)  # (batch_size, 2048, image_size/32, image_size/32)

        out = self.model.extract_features(images)
        # #print("image size", out.size())

        # # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = self.adaptive_pool(out)
        # # (batch_size, encoded_image_size, encoded_image_size, 2048)
        # # (later on the intermidiate dims are flatten: (prepare_inputs)
        # # (batch_size, encoded_image_size*encoded_image_size, 2048)
        encoder_features = out.permute(0, 2, 3, 1)
        encoder_attrs = self.model(images)

        return encoder_features, encoder_attrs

    def fine_tune(self, enable_fine_tuning):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        if enable_fine_tuning:
            print("fine-tuning encoder")
            if self.model_type == ImageNetModelsPretrained.MULTILABEL_ALL_EFFICIENCENET.value:
                for c in list(self.model.children())[-6:-5]:  # [-6:-1]
                    print("unfreezing eff, layer", c)
                    for p in c.parameters():
                        p.requires_grad = enable_fine_tuning

            elif self.model_type == ImageNetModelsPretrained.EFFICIENCENET_EMBEDDINGS.value:
                for c in list(self.model.children())[-1:]:  # [5:]:#toda!!!
                    print("unfreezing layer:", c)
                    for p in c.parameters():
                        p.requires_grad = enable_fine_tuning
            else:  # All layers
                for c in list(self.model.children()):
                    print("unfreezing layer", c)
                    for p in c.parameters():
                        p.requires_grad = enable_fine_tuning


class ContinuousDecoderWithAttention(DecoderWithAttention):
    """
    Decoder.
    """

    def __init__(
            self, attention_dim, embedding_type, embed_dim, decoder_dim, vocab_size, token_to_id, post_processing,
            encoder_dim=2048, dropout=0.5):

        super(ContinuousDecoderWithAttention, self).__init__(attention_dim, embedding_type,
                                                             embed_dim, decoder_dim, vocab_size, token_to_id, post_processing, encoder_dim, dropout)

        # replace softmax layer with embedding layer
        self.fc = nn.Linear(decoder_dim, embed_dim)

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        #c = self.init_c(mean_encoder_out)
        c = h
        return h, c

class ContinuousScaleProductAttention3CompGradNormModel(ContinuousEncoderDecoderModel):

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

        # if (self.args.embedding_type not in [embedding.value for embedding in EmbeddingsType]):
        #     raise ValueError(
        #         "Continuous model should use pretrained embeddings...")

        self.encoder = Encoder(self.args.image_model_type,
                               enable_fine_tuning=self.args.fine_tune_encoder)

        self.decoder = ContinuousDecoderWithAttention(
            encoder_dim=self.encoder.encoder_dim,
            attention_dim=self.args.attention_dim,
            decoder_dim=self.args.decoder_dim,
            embedding_type=self.args.embedding_type,
            embed_dim=self.args.embed_dim,
            vocab_size=self.vocab_size,
            token_to_id=self.token_to_id,
            post_processing=self.args.post_processing,
            dropout=self.args.dropout
        )

        self.decoder.normalize_embeddings(self.args.no_normalization)

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        if self.args.grad_norm:

            self.loss_weight_word = torch.ones(
                1, requires_grad=True, device=self.device, dtype=torch.float
            )
            self.loss_weight_sent = torch.ones(
                1, requires_grad=True, device=self.device, dtype=torch.float
            )

            self.loss_weight_input1 = torch.ones(
                1, requires_grad=True, device=self.device, dtype=torch.float
            )

            self.gradnorm_optimizer = torch.optim.Adam(
                [self.loss_weight_word, self.loss_weight_sent, self.loss_weight_input1],
                lr=0.025,
            )
            self.gradnorm_loss = nn.L1Loss().to(self.device)
        else:
            self.loss_weight_word = torch.ones(
                1, device=self.device, dtype=torch.float
            )*self.args.w1

            self.loss_weight_sent = torch.ones(
                1, device=self.device, dtype=torch.float
            )*self.args.w2

            self.loss_weight_input1 = torch.ones(
                1, device=self.device, dtype=torch.float
            )*self.args.w3

        self.initial = False
        self.decodying_criteria = torch.nn.SmoothL1Loss(reduction="none")

    def train(self, train_dataloader, val_dataloader, print_freq):
        if self.args.early_mode == "loss":
            start_baseline_value = np.Inf
        elif self.args.early_mode == "metric":
            start_baseline_value = 0

        early_stopping = EarlyStopping(
            epochs_limit_without_improvement=self.args.epochs_limit_without_improvement,
            epochs_since_last_improvement=self.checkpoint_epochs_since_last_improvement
            if self.checkpoint_exists else 0,
            baseline=self.checkpoint_val_loss if self.checkpoint_exists else start_baseline_value,
            encoder_optimizer=self.encoder_optimizer,
            decoder_optimizer=self.decoder_optimizer,
            period_decay_lr=self.args.period_decay_without_improvement,
            mode=self.args.early_mode
        )

        start_epoch = self.checkpoint_start_epoch if self.checkpoint_exists else 0
        all_training_losses = []
        all_validation_loss = []
        all_validation_bleu = []
        all_val_best=[]
        all_w_word=[]
        all_w_sent=[]
        all_w_input1=[]
        all_together=[]

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

                train_total_loss += train_loss.data.item()
                del train_loss

                # (only for debug: interrupt val after 1 step)
                if self.args.disable_steps:
                    break

            # End training
            epoch_loss = train_total_loss / (batch_i + 1)
            all_training_losses.append(epoch_loss)
            logging.info('Time taken for 1 epoch {:.4f} sec'.format(
                time.time() - start))
            logging.info('\n\n-----> TRAIN END! Epoch: {}; Loss: {:.4f}\n'.format(epoch,
                                                                                  train_total_loss / (batch_i + 1)))

            # Start validation
            self.decoder.eval()  # eval mode (no dropout or batchnorm)
            self.encoder.eval()

            with torch.no_grad():
                all_hypotheses = []
                all_references = []
                for batch_i, (imgs, caps, caplens, all_captions) in enumerate(val_dataloader):

                    val_loss, val_hypotheses, val_references = self.val_step(
                        imgs, caps, caplens, all_captions)

                    all_hypotheses.extend(val_hypotheses)
                    all_references.extend(val_references)
                    #print("val hy", val_hypotheses)
                    #print("val references", val_references)

                    self._log_status("VAL", epoch, batch_i,
                                     val_dataloader, val_loss, print_freq)

                    val_total_loss += val_loss

                    # (only for debug: interrupt val after 1 step)
                    if self.args.disable_steps:
                        break

            # End validation
            epoch_val_loss = val_total_loss / (batch_i + 1)
            all_validation_loss.append(epoch_val_loss.item())

            if self.args.early_mode == "loss":
                epoch_val_score = epoch_val_loss
                epoch_val_bleu4 = -1.0
            elif self.args.early_mode == "metric":
                epoch_val_bleu4 = corpus_bleu(all_references, all_hypotheses)
                all_validation_bleu.append(epoch_val_bleu4)
                epoch_val_score = epoch_val_bleu4

            early_stopping.check_improvement(epoch_val_score)
            self._save_checkpoint(early_stopping.is_current_val_best(),
                                  epoch,
                                  early_stopping.get_number_of_epochs_without_improvement(),
                                  epoch_val_score)

            logging.info('\n-------------- END EPOCH:{}⁄{}; Train Loss:{:.4f}; Val Loss:{:.4f}; Val Bleu:{:.3f}; -------------\n'.format(
                epoch, self.args.epochs, epoch_loss, epoch_val_loss, epoch_val_bleu4))

            all_val_best.append(early_stopping.is_current_val_best())
            all_w_word.append(self.loss_weight_word[0].item())
            all_w_sent.append(self.loss_weight_sent[0].item())
            all_w_input1.append(self.loss_weight_input1[0].item())
            all_together.append((epoch_loss, epoch_val_loss.item(), early_stopping.is_current_val_best(), self.loss_weight_word[0].item(),self.loss_weight_sent[0].item(), self.loss_weight_input1[0].item()) )

        final_dict = {
            "train_loss": all_training_losses,
            "val_loss": all_validation_loss,
            "val_bleu": all_validation_bleu,
            "all_val_best":all_val_best,
            "all_w_word":all_w_word,
            "all_w_sent":all_w_sent,
            "all_w_input1":all_w_input1,
            "all_together": all_together,
        }

        with open(PATH_TRAINED_MODELS + self.args.file_name + ".json", 'w+') as f:
            json.dump(final_dict, f, indent=2)

    def _prepare_inputs_to_forward_pass(self, imgs, caps, caption_lengths):
        imgs = imgs.to(self.device)
        caps = caps.to(self.device)
        caption_lengths = caption_lengths.to(self.device)

        encoder_features, encoder_attrs = self.encoder(imgs)
        encoder_features = encoder_features.view(
            encoder_features.size(0), -1, encoder_features.size(-1))  # flatten

        # sorted captions
        caption_lengths, sort_ind = caption_lengths.squeeze(
            1).sort(dim=0, descending=True)
        encoder_features = encoder_features[sort_ind]
        encoder_attrs = encoder_attrs[sort_ind]
        caps_sorted = caps[sort_ind]

        # input captions must not have "end_token"
        caption_lengths = (caption_lengths - 1).tolist()
        encoder_out = encoder_features, encoder_attrs

        return encoder_out, caps_sorted, caption_lengths, sort_ind

    def _predict(self, encoder_results, caps, caption_lengths):
        encoder_out, encoder_attrs = encoder_results
        
        self.decoder.image_embedding = torch.nn.functional.normalize(encoder_attrs, p=2, dim=-1)

        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)

        # Create tensors to hold word predicion scores and alphas
        all_predictions = torch.zeros(batch_size, max(
            caption_lengths), self.decoder.embed_dim).to(self.device)
        all_alphas = torch.zeros(batch_size, max(
            caption_lengths), num_pixels).to(self.device)

        h, c = self.decoder.init_hidden_state(encoder_out)

        # Predict
        for t in range(max(
                caption_lengths)):
            # batchsizes of current time_step are the ones with lenght bigger than time-step (i.e have not fineshed yet)
            batch_size_t = sum([l > t for l in caption_lengths])

            predictions, h, c, alpha = self.decoder(
                caps[:batch_size_t, t], encoder_out[:batch_size_t], h[:batch_size_t], c[:batch_size_t])

            all_predictions[:batch_size_t, t, :] = predictions
            all_alphas[:batch_size_t, t, :] = alpha

        return {"predictions": all_predictions, "alphas": all_alphas}

    def _calculate_loss(self, predict_output, caps, caption_lengths):
        predictions = predict_output["predictions"]
        targets = caps[:, 1:]  # targets doesnt have stark token

        target_embeddings = self.decoder.embedding(targets).to(self.device)

        if self.args.no_normalization == False:
            # when target embeddings start normalized, predictions should also be normalized
            predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)

        word_loss, sentence_loss, input1_loss = self.loss_method(
            predictions,
            target_embeddings,
            caption_lengths
        )

        return word_loss, sentence_loss, input1_loss   

    def _define_loss_criteria(self):
        if self.args.continuous_loss_type == ContinuousLossesType.SMOOTHL1_REAL_AVG_SENTENCE_AND_INPUT.value:
            self.loss_method = self.smoothl1_real_avg_sentence_and_input_loss
            self.criterion = nn.SmoothL1Loss(reduction='none').to(self.device)
        else:
            raise Exception("invalid loss")

    def smoothl1_real_avg_sentence_and_input_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths
    ):
        word_losses = 0.0  # pred_against_target_loss; #pred_sentence_again_target_sentence;"pred_sentence_agains_image
        sentence_losses = 0.0
        input1_losses = 0.0

        images_embedding = self.decoder.image_embedding

        n_sentences = predictions.size()[0]
        for i in range(n_sentences):  # iterate by sentence
            preds_without_padd = predictions[i, :caption_lengths[i], :]
            targets_without_padd = target_embeddings[i, :caption_lengths[i], :]

            # word-level loss   (each prediction against each target)
            w_loss = self.criterion(preds_without_padd, targets_without_padd)            
            w_loss = torch.sum(w_loss, dim=-1)
            w_loss = torch.mean(w_loss)

            word_losses += w_loss

            # sentence-level loss (sentence predicted agains target sentence)
            sentence_mean_pred = torch.mean(preds_without_padd, dim=0).unsqueeze(0)  # ver a dim
            sentece_mean_target = torch.mean(targets_without_padd, dim=0).unsqueeze(0)

            s_loss = self.criterion(sentence_mean_pred, sentece_mean_target)
            s_loss = torch.sum(s_loss, dim=-1)
            s_loss = torch.mean(s_loss)

            sentence_losses += s_loss

            # 1º input loss (sentence predicted against input image)
            image_embedding = images_embedding[i].unsqueeze(0)

            i_loss = self.criterion(sentence_mean_pred, image_embedding)
            i_loss = torch.sum(i_loss, dim=-1)
            i_loss = torch.mean(i_loss)
            input1_losses += i_loss

        word_loss = word_losses / n_sentences
        sentence_loss = sentence_losses / n_sentences
        input1_loss = input1_losses / n_sentences

        return word_loss, sentence_loss, input1_loss

    def val_step(self, imgs, caps_input, cap_len, all_captions):
        (loss_word, loss_sent, loss_input1), hypotheses, references_without_padding = super().val_step(
            imgs, caps_input, cap_len, all_captions)
        loss = self.loss_weight_word[0].data * loss_word +\
            self.loss_weight_sent[0].data * loss_sent + \
            self.loss_weight_input1[0].data * loss_input1

        print("weight word", self.loss_weight_word[0].data)
        print("weight sent", self.loss_weight_sent[0].data)
        print("weight intput1", self.loss_weight_input1[0].data)

        return loss, hypotheses, references_without_padding

    def train_step(self, imgs, caps_input, cap_len):
        encoder_out, caps_sorted, caption_lengths, sort_ind = self._prepare_inputs_to_forward_pass(
            imgs, caps_input, cap_len)

        predict_output = self._predict(
            encoder_out, caps_sorted, caption_lengths)

        loss_word, loss_sent, loss_input1 = self._calculate_loss(
            predict_output, caps_sorted, caption_lengths)

        if self.initial == False:
            self.initial = True
            self.initial_word_loss = loss_word
            self.initial_sent_loss = loss_sent
            self.initial_input1_loss = loss_input1

        loss = self.loss_weight_word[0] * loss_word + \
            self.loss_weight_sent[0] * loss_sent + \
            self.loss_weight_input1[0] * loss_input1

        self.decoder_optimizer.zero_grad()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()

        loss.backward(retain_graph=self.args.grad_norm)

        if self.args.grad_norm:
            self.apply_grad_norm(loss_word, loss_sent, loss_input1)

        # # Clip gradients
        clip_gradient(self.decoder_optimizer, 5.)
        if self.encoder_optimizer is not None:
            clip_gradient(self.encoder_optimizer, 5.)

        # Update weights
        self.decoder_optimizer.step()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.step()

        return loss

    def apply_grad_norm(self, loss_word, loss_sent, loss_input1):

        G1R = torch.autograd.grad(
            loss_word, self.decoder.fc.parameters(), retain_graph=True, create_graph=True
        )
        G1R_flattened = torch.cat([g.view(-1) for g in G1R])
        G1 = torch.norm(self.loss_weight_word * G1R_flattened.detach(), 2).unsqueeze(0)

        G2R = torch.autograd.grad(loss_sent, self.decoder.fc.parameters(), retain_graph=True)
        G2R_flattened = torch.cat([g.view(-1) for g in G2R])
        G2 = torch.norm(self.loss_weight_sent * G2R_flattened.detach(), 2).unsqueeze(0)

        G3R = torch.autograd.grad(loss_input1, self.decoder.fc.parameters(), retain_graph=True)
        G3R_flattened = torch.cat([g.view(-1) for g in G3R])
        G3 = torch.norm(self.loss_weight_input1 * G3R_flattened.detach(), 2).unsqueeze(0)

        # Calculate the average gradient norm across all tasks
        G_avg = torch.div(G1 + G2 + G3, 3)

        # Calculate relative losses
        lhat1 = torch.div(loss_word.detach(), self.initial_word_loss)
        lhat2 = torch.div(loss_sent.detach(), self.initial_sent_loss)
        lhat3 = torch.div(loss_input1.detach(), self.initial_input1_loss)

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

        coef = 3 / (self.loss_weight_word + self.loss_weight_sent + self.loss_weight_input1)
        self.loss_weight_word.data = coef.data * self.loss_weight_word.data
        self.loss_weight_sent.data = coef.data * self.loss_weight_sent.data
        self.loss_weight_input1.data = coef.data * self.loss_weight_input1.data


    def inference_with_greedy_smoothl1(self, image, n_solutions=0, min_len=0, repetition_window=0, max_len=50):
        with torch.no_grad():  # no need to track history

            decoder_sentence = []

            input_word = torch.tensor([self.token_to_id[START_TOKEN]])

            i = 1

            encoder_output, encoder_attrs = self.encoder(image)
            self.decoder.image_embedding = torch.nn.functional.normalize(encoder_attrs, p=2, dim=-1)

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

    def inference_with_greedy_smoothl1_no_reps(self, image, n_solutions=0, min_len=0, repetition_window=0, max_len=50):
        with torch.no_grad():  # no need to track history

            decoder_sentence = []

            input_word = torch.tensor([self.token_to_id[START_TOKEN]])

            i = 1

            encoder_output, encoder_attrs = self.encoder(image)
            self.decoder.image_embedding = torch.nn.functional.normalize(encoder_attrs, p=2, dim=-1)

            encoder_output = encoder_output.view(
                1, -1, encoder_output.size()[-1])

            h, c = self.decoder.init_hidden_state(encoder_output)

            criteria = torch.nn.SmoothL1Loss(reduction="none")

            previous_output_token = ""

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

                if current_output_token == previous_output_token:
                    current_output_index = sorted_indices.squeeze()[1]
                    current_output_token = self.id_to_token[current_output_index.item(
                    )]

                if current_output_token == END_TOKEN and i<=min_len: #sentences with min len
                    current_output_index = sorted_indices.squeeze()[1]
                    current_output_token = self.id_to_token[current_output_index.item(
                    )]

                previous_output_token = current_output_token

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

    def inference_with_greedy_smoothl1_mmr(self, image, n_solutions=0, min_len=0, repetition_window=0, max_len=50):
        alpha=0.95
        with torch.no_grad():  # no need to track history

            decoder_sentence = []

            input_word = torch.tensor([self.token_to_id[START_TOKEN]])

            i = 1

            encoder_output, encoder_attrs = self.encoder(image)
            self.decoder.image_embedding = torch.nn.functional.normalize(encoder_attrs, p=2, dim=-1)

            encoder_output = encoder_output.view(
                1, -1, encoder_output.size()[-1])

            h, c = self.decoder.init_hidden_state(encoder_output)

            criteria = torch.nn.SmoothL1Loss(reduction="none")

            all_prev_tokens=self.decoder.embedding(torch.tensor([self.token_to_id[START_TOKEN]]))

            while True:

                scores, h, c = self.generate_output_index_smoothl1(criteria,
                                                                   input_word, encoder_output, h, c)

                #embeddings against previous generated words -> to have a score of diversity
                n_prev_tokens= len(all_prev_tokens)
                scores_second_part = torch.zeros(len(scores), n_prev_tokens) 
                #print("scores before", scores_second_part.size())

                for j in range(n_prev_tokens):
                    #print("mean1", criteria(self.decoder.embedding.weight.data, all_prev_tokens[j].expand_as(self.decoder.embedding.weight.data)).mean(1))
                    scores_second_part[:, j] = criteria(self.decoder.embedding.weight.data, all_prev_tokens[j].expand_as(self.decoder.embedding.weight.data)).mean(1)
                
                # print("scores after", scores_second_part)
                # print("scores just first", scores_second_part[0,:])

                #scores_second_part = torch.clamp(scores_second_part, min=0)
                scores_diversity, diversity_index= torch.min(scores_second_part, dim=-1)
                # print("scores diversity size", scores_diversity.size())
                # print("scores_diversity", scores_diversity)
                # print("scores_diversity 0", scores_diversity[0])

                # scores = alpha*scores - (1-alpha)*scores_diversity
                # print("scores mmr", scores.size())

                # sorted_scores, sorted_indices = torch.sort(scores, descending=False, dim=-1)
                # print("sorted ind", sorted_indices)
                # print("sorted values", scores)

                # print("mmr", alpha*scores - (1-alpha)*scores_diversity)
                sorted_scores, sorted_indices = torch.sort(alpha*scores + (1-alpha)*scores_diversity, descending=False, dim=-1)
                #print("sorted ind mmr", sorted_indices)
                #print(stop)

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

                if current_output_token == END_TOKEN and i<=min_len: #sentences with min len
                    current_output_index = sorted_indices.squeeze()[1]
                    current_output_token = self.id_to_token[current_output_index.item(
                    )]

                all_prev_tokens = torch.cat((all_prev_tokens, self.decoder.embedding(torch.tensor([current_output_index.item(
                )]))), 0) 

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

    def generate_output_index_smoothl1(self, criteria, input_word, encoder_out, h, c):
        predictions, h, c,_ = self.decoder(input_word, encoder_out, h, c)

        current_output_index = self._convert_prediction_to_output_smoothl1(criteria, predictions)

        return current_output_index, h, c   

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
            scores, h, c = self.generate_output_index_smoothl1(self.decodying_criteria,
                torch.tensor([self.token_to_id[last_token]]), encoder_out, h, c)

            sorted_scores, sorted_indices = torch.sort(
                scores.squeeze(), descending=False, dim=-1)

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
            return sorted(candidates, key=operator.itemgetter(1), reverse=False)[:n_solutions]

        with torch.no_grad():
            my_dict = {}

            encoder_output, encoder_attrs = self.encoder(image)
            self.decoder.image_embedding = torch.nn.functional.normalize(encoder_attrs, p=2, dim=-1)

            encoder_output = encoder_output.view(
                1, -1, encoder_output.size()[-1])

            # encoder_output = self.encoder(image.to(self.device))
            # encoder_output = encoder_output.view(1, -1, encoder_output.size()[-1])  # flatten encoder
            h, c = self.decoder.init_hidden_state(encoder_output)

            top_solutions = [([START_TOKEN], 0.0, h, c)]

            for time_step in range(self.max_len - 1):
                # print("\nnew time step")
                candidates = []
                for sentence, prob, h, c in top_solutions:
                    candidates.extend(generate_n_solutions(
                        sentence, prob, encoder_output, h, c, n_solutions))

                top_solutions = get_most_probable(candidates, n_solutions)


            best_tokens, prob, h, c = top_solutions[0]

            if best_tokens[0] == START_TOKEN:
                best_tokens = best_tokens[1:]
            if best_tokens[-1] == END_TOKEN:
                best_tokens = best_tokens[:-1]
            best_sentence = " ".join(best_tokens)

            print("\nbeam decoded sentence:", best_sentence)
            return best_sentence

    def generate_output_index(self, input_word, encoder_out, h, c):
        current_output_index, h, c = self.generate_output_index_smoothl1(self.decodying_criteria, input_word, encoder_out, h, c)
        return current_output_index, h, c   
