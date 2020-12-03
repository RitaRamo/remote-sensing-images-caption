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

        if (self.args.embedding_type not in [embedding.value for embedding in EmbeddingsType]):
            raise ValueError(
                "Continuous model should use pretrained embeddings...")

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
            )
            self.loss_weight_sent = torch.ones(
                1, device=self.device, dtype=torch.float
            )
            self.loss_weight_input1 = torch.ones(
                1, device=self.device, dtype=torch.float
            )

        self.initial = False

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

    def generate_output_index_smoothl1(self, criteria, input_word, encoder_out, h, c):
        predictions, h, c,_ = self.decoder(
            input_word, encoder_out, h, c)

        current_output_index = self._convert_prediction_to_output_smoothl1(criteria, predictions)

        return current_output_index, h, c   