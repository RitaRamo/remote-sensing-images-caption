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


class ContinuousDecoderWithImage(Decoder):

    def __init__(self, decoder_dim, embed_dim, embedding_type, vocab_size, token_to_id, post_processing,
                 encoder_dim=2048, dropout=0.5):

        super(ContinuousDecoderWithImage, self).__init__(decoder_dim, embed_dim,
                                                         embedding_type, vocab_size, token_to_id, post_processing, encoder_dim, dropout)

        # linear layer to find representation of image
        self.represent_image = nn.Linear(encoder_dim, embed_dim)
        self.image_embedding = None

        # replace softmax with a embedding layer
        self.fc = nn.Linear(decoder_dim, embed_dim)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)

        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        self.image_embedding = self.represent_image(mean_encoder_out)

        return h, h


class ContinuousEncoderDecoderGradNormImageModel(ContinuousEncoderDecoderModel):

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

        self.decoder = ContinuousDecoderWithImage(
            encoder_dim=self.encoder.encoder_dim,
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

            self.loss_weight_input2 = torch.ones(
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
            )
            self.loss_weight_sent = torch.ones(
                1, device=self.device, dtype=torch.float
            )
            self.loss_weight_input1 = torch.ones(
                1, device=self.device, dtype=torch.float
            )
            self.loss_weight_input2 = torch.ones(
                1, device=self.device, dtype=torch.float
            )
        self.initial = False
        print("AQUI loss weights word", self.loss_weight_word.item())
        print("AWUI loss weights sent", self.loss_weight_sent.item())
        print("AWUI loss weights sent", self.loss_weight_input1.item())
        print("AWUI loss weights sent", self.loss_weight_input2.item())

    def _define_loss_criteria(self):
        if self.args.continuous_loss_type == "cos_avg_sentence_and_inputs_loss":
            self.loss_method = self.cos_avg_sentence_and_inputs_loss
            self.criterion = nn.CosineEmbeddingLoss().to(self.device)
        else:
            raise Exception("invalid loss")

    def cos_avg_sentence_and_inputs_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths
    ):
        word_losses = 0.0  # pred_against_target_loss; #pred_sentence_again_target_sentence;"pred_sentence_agains_image
        sentence_losses = 0.0
        input1_losses = 0.0
        input2_losses = 0.0

        images_embedding = self.decoder.image_embedding

        n_sentences = predictions.size()[0]
        for i in range(n_sentences):  # iterate by sentence
            preds_without_padd = predictions[i, :caption_lengths[i], :]
            targets_without_padd = target_embeddings[i, :caption_lengths[i], :]
            y = torch.ones(targets_without_padd.shape[0]).to(self.device)

            # word-level loss   (each prediction against each target)
            word_losses += self.criterion(
                preds_without_padd,
                targets_without_padd,
                y
            )

            # sentence-level loss (sentence predicted agains target sentence)
            sentence_mean_pred = torch.mean(preds_without_padd, dim=0).unsqueeze(0)  # ver a dim
            sentece_mean_target = torch.mean(targets_without_padd, dim=0).unsqueeze(0)

            y = torch.ones(1).to(self.device)

            sentence_losses += self.criterion(
                sentence_mean_pred,
                sentece_mean_target,
                y
            )

            image_embedding = images_embedding[i].unsqueeze(0)

            # 1º input loss (sentence predicted against input image)
            input1_losses += self.criterion(
                sentence_mean_pred,
                image_embedding,
                y
            )

            # 2º input loss (sentence predicted against input image)
            input2_losses += self.criterion(
                image_embedding,
                sentece_mean_target,
                y
            )

        word_loss = word_losses / n_sentences
        sentence_loss = sentence_losses / n_sentences
        input1_loss = input1_losses / n_sentences
        input2_loss = input2_losses / n_sentences

        if self.initial == False:
            self.initial = True
            self.initial_word_loss = word_loss
            self.initial_sent_loss = sentence_loss
            self.initial_input1_loss = input1_loss
            self.initial_input2_loss = input2_loss

        return word_loss, sentence_loss, input1_loss, input2_loss

    def _calculate_loss(self, predict_output, caps, caption_lengths):
        predictions = predict_output["predictions"]
        targets = caps[:, 1:]  # targets doesnt have stark token

        target_embeddings = self.decoder.embedding(targets).to(self.device)

        if self.args.no_normalization == False:
            # when target embeddings start normalized, predictions should also be normalized
            predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)

        word_loss, sentence_loss, input1_loss, input2_loss = self.loss_method(
            predictions,
            target_embeddings,
            caption_lengths
        )

        return word_loss, sentence_loss, input1_loss, input2_loss

    def val_step(self, imgs, caps_input, cap_len, all_captions):
        (loss_word, loss_sent, loss_input1, loss_input2), hypotheses, references_without_padding = super().val_step(
            imgs, caps_input, cap_len, all_captions)
        loss = self.loss_weight_word[0].data * loss_word +\
            self.loss_weight_sent[0].data * loss_sent + \
            self.loss_weight_input1[0].data * loss_input1 + \
            self.loss_weight_input2[0].data * loss_input2

        print("weight word", self.loss_weight_word[0].data)
        print("weight sent", self.loss_weight_sent[0].data)
        print("weight intput1", self.loss_weight_input1[0].data)
        print("weight input2", self.loss_weight_input2[0].data)

        return loss, hypotheses, references_without_padding

    def train_step(self, imgs, caps_input, cap_len):
        encoder_out, caps_sorted, caption_lengths, sort_ind = self._prepare_inputs_to_forward_pass(
            imgs, caps_input, cap_len)

        predict_output = self._predict(
            encoder_out, caps_sorted, caption_lengths)

        loss_word, loss_sent, loss_input1, loss_input2 = self._calculate_loss(
            predict_output, caps_sorted, caption_lengths)

        loss = self.loss_weight_word[0] * loss_word + \
            self.loss_weight_sent[0] * loss_sent + \
            self.loss_weight_input1[0] * loss_input1 + \
            self.loss_weight_input2[0] * loss_input2

        self.decoder_optimizer.zero_grad()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()

        loss.backward(retain_graph=self.args.grad_norm)

        if self.args.grad_norm:
            self.apply_grad_norm(loss_word, loss_sent, loss_input1, loss_input2)

        # # Clip gradients
        clip_gradient(self.decoder_optimizer, 5.)
        if self.encoder_optimizer is not None:
            clip_gradient(self.encoder_optimizer, 5.)

        # Update weights
        self.decoder_optimizer.step()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.step()

        return loss

    def apply_grad_norm(self, loss_word, loss_sent, loss_input1, loss_input2):

        # shared_params = [
        #     param for param in self.decoder.parameters() if param.requires_grad
        # ]

        SHARED_PARAMS = [
            "represent_image.weight",
            "represent_image.bias"
        ]

        named_params = dict(self.decoder.named_parameters())

        shared_params = [
            param
            for param_name, param in named_params.items()
            if param_name in SHARED_PARAMS and param.requires_grad
        ]

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

        G4R = torch.autograd.grad(loss_input2, shared_params)
        G4R_flattened = torch.cat([g.view(-1) for g in G4R])
        G4 = torch.norm(self.loss_weight_input2 * G4R_flattened.detach(), 2).unsqueeze(0)
        # Calculate the average gradient norm across all tasks
        G_avg = torch.div(G1 + G2 + G3 + G4, 4)

        # Calculate relative losses
        lhat1 = torch.div(loss_word.detach(), self.initial_word_loss)
        lhat2 = torch.div(loss_sent.detach(), self.initial_sent_loss)
        lhat3 = torch.div(loss_input1.detach(), self.initial_input1_loss)
        lhat4 = torch.div(loss_input2.detach(), self.initial_input2_loss)

        lhat_avg = torch.div(lhat1 + lhat2 + lhat3 + lhat4, 4)

        # Calculate relative inverse training rates
        inv_rate1 = torch.div(lhat1, lhat_avg)
        inv_rate2 = torch.div(lhat2, lhat_avg)
        inv_rate3 = torch.div(lhat3, lhat_avg)
        inv_rate4 = torch.div(lhat4, lhat_avg)

        # Calculate the gradient norm target for this batch
        C1 = G_avg * (inv_rate1 ** self.args.grad_norm_alpha)
        C2 = G_avg * (inv_rate2 ** self.args.grad_norm_alpha)
        C3 = G_avg * (inv_rate3 ** self.args.grad_norm_alpha)
        C4 = G_avg * (inv_rate4 ** self.args.grad_norm_alpha)

        C1 = C1.detach()
        C2 = C2.detach()
        C3 = C3.detach()
        C4 = C4.detach()

        # Backprop and perform an optimization step
        self.gradnorm_optimizer.zero_grad()
        # Calculate the gradnorm loss
        Lgrad = self.gradnorm_loss(G1, C1) + self.gradnorm_loss(G2,
                                                                C2) + self.gradnorm_loss(G3, C3) + self.gradnorm_loss(G4, C4)
        Lgrad.backward()
        self.gradnorm_optimizer.step()

        coef = 4 / (self.loss_weight_word + self.loss_weight_sent + self.loss_weight_input1 + self.loss_weight_input2)
        self.loss_weight_word.data = coef.data * self.loss_weight_word.data
        self.loss_weight_sent.data = coef.data * self.loss_weight_sent.data
        self.loss_weight_input1.data = coef.data * self.loss_weight_input1.data
        self.loss_weight_input2.data = coef.data * self.loss_weight_input2.data
