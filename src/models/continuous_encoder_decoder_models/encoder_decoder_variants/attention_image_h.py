import torchvision
from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from models.basic_encoder_decoder_models.encoder_decoder_variants.attention import Attention, Encoder, DecoderWithAttention
from models.abtract_model import AbstractEncoderDecoderModel
import torch.nn.functional as F
from embeddings.embeddings import get_embedding_layer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from data_preprocessing.preprocess_tokens import OOV_TOKEN
from embeddings.embeddings import EmbeddingsType
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention import ContinuousAttentionModel
from embeddings.embeddings import EmbeddingsType
from data_preprocessing.preprocess_tokens import START_TOKEN, END_TOKEN
from utils.enums import ContinuousLossesType


class ContinuousDecoderWithAttentionAndImage(DecoderWithAttention):
    """
    Decoder.
    """

    def __init__(
            self, attention_dim, embedding_type, embed_dim, decoder_dim, vocab_size, token_to_id, post_processing,
            encoder_dim=2048, dropout=0.5):

        super(ContinuousDecoderWithAttentionAndImage, self).__init__(attention_dim, embedding_type,
                                                                     embed_dim, decoder_dim, vocab_size, token_to_id, post_processing, encoder_dim, dropout)

        # linear layer to find representation of image
        self.represent_image = nn.Linear(encoder_dim, embed_dim)
        self.image_embedding = None

        # replace softmax layer with embedding layer
        self.fc = nn.Linear(decoder_dim, embed_dim)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)

        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim) 512 units

        self.image_embedding = self.represent_image(mean_encoder_out)  # 300 512

        return h, h

    def inference(self, embeddings, encoder_out, decoder_hidden_state, decoder_cell_state):
        attention_weighted_encoding, alpha = self.attention(encoder_out, decoder_hidden_state)

        decoder_input = torch.cat((embeddings, attention_weighted_encoding), dim=1)

        decoder_hidden_state, decoder_cell_state = self.decode_step(
            decoder_input, (decoder_hidden_state, decoder_cell_state)
        )

        scores = self.fc(self.dropout(decoder_hidden_state))

        return scores, decoder_hidden_state, decoder_cell_state, alpha


class ContinuousAttentionImageHModel(ContinuousAttentionModel):

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

        self.decoder = ContinuousDecoderWithAttentionAndImage(
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

    def generate_output_embedding(self, input_embedding, encoder_out, h, c):
        predictions, h, c, _ = self.decoder.inference(
            input_embedding, encoder_out, h, c)

        current_output_index = self._convert_prediction_to_output(predictions)

        return predictions, current_output_index, h, c

    def inference_with_greedy_embedding(self, image):
        with torch.no_grad():  # no need to track history

            decoder_sentence = START_TOKEN + " "

            input_embedding = self.decoder.embedding(torch.tensor([self.token_to_id[START_TOKEN]]))

            i = 1

            encoder_output = self.encoder(image)
            encoder_output = encoder_output.view(
                1, -1, encoder_output.size()[-1])

            h, c = self.decoder.init_hidden_state(encoder_output)

            while True:

                predictions, scores, h, c = self.generate_output_embedding(
                    input_embedding, encoder_output, h, c)

                sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)

                current_output_index = sorted_indices[0]

                current_output_token = self.id_to_token[current_output_index.item(
                )]

                decoder_sentence += " " + current_output_token

                if (current_output_token == END_TOKEN or
                        i >= self.max_len - 1):  # until 35
                    break

                input_embedding[0, :] = predictions

                i += 1

            print("\ndecoded sentence", decoder_sentence)

            return decoder_sentence  # input_caption

    def inference_with_greedy_smoothl1(self, image):
        with torch.no_grad():  # no need to track history
            self.criterion = nn.SmoothL1Loss(reduction='none').to(self.device)

            decoder_sentence = START_TOKEN + " "

            input_word = torch.tensor([self.token_to_id[START_TOKEN]])

            i = 1

            encoder_output = self.encoder(image)
            encoder_output = encoder_output.view(
                1, -1, encoder_output.size()[-1])

            h, c = self.decoder.init_hidden_state(encoder_output)

            while True:

                predictions, h, c, _ = self.decoder(input_word, encoder_output, h, c)
                scores = self.criterion(predictions, self.decoder.embedding.weight.data)
                scores = torch.mean(scores, dim=1)

                sorted_scores, sorted_indices = torch.sort(scores, descending=False, dim=-1)

                current_output_index = sorted_indices[0]

                current_output_token = self.id_to_token[current_output_index.item(
                )]

                decoder_sentence += " " + current_output_token

                if (current_output_token == END_TOKEN or
                        i >= self.max_len - 1):  # until 35
                    break

                input_word[0] = current_output_index.item()

                i += 1

            print("\ndecoded sentence", decoder_sentence)

            return decoder_sentence  # input_caption

    def _predict(self, encoder_out, caps, caption_lengths):
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)

        # Create tensors to hold word predicion scores and alphas
        all_predictions = torch.zeros(batch_size, max(
            caption_lengths), self.decoder.embed_dim).to(self.device)
        all_alphas = torch.zeros(batch_size, max(
            caption_lengths), num_pixels).to(self.device)
        all_hs = torch.zeros(batch_size, max(
            caption_lengths), self.decoder.decoder_dim).to(self.device)

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
            all_hs[:batch_size_t, t, :] = h

        return {"predictions": all_predictions, "alphas": all_alphas, "hs": all_hs}

    def _calculate_loss(self, predict_output, caps, caption_lengths):
        predictions = predict_output["predictions"]
        targets = caps[:, 1:]  # targets doesnt have stark token
        hs = predict_output["hs"]

        target_embeddings = self.decoder.embedding(targets).to(self.device)

        if self.args.no_normalization == False:
            # when target embeddings start normalized, predictions should also be normalized
            predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)

        loss = self.loss_method(
            predictions,
            target_embeddings,
            hs,
            caption_lengths,
        )

        return loss

    def _define_loss_criteria(self):
        loss_type = self.args.continuous_loss_type

        if loss_type == ContinuousLossesType.COS_13.value:
            self.loss_method = self.cos_and_third_h_loss
            self.criterion = nn.CosineEmbeddingLoss().to(self.device)

        else:
            raise Exception("only available: cos_13 ")

    def cos_and_third_h_loss(
        self,
        predictions,
        target_embeddings,
        hs,
        caption_lengths
    ):
        word_losses = 0.0  # pred_against_target_loss; #pred_sentence_again_target_sentence;"pred_sentence_agains_image
        input1_losses = 0.0

        images_embedding = self.decoder.image_embedding

        n_sentences = predictions.size()[0]
        for i in range(n_sentences):  # iterate by sentence
            preds_without_padd = predictions[i, :caption_lengths[i], :]
            targets_without_padd = target_embeddings[i, :caption_lengths[i], :]
            final_hs = hs[i, caption_lengths[i] - 1, :].unsqueeze(0)  # get last h

            y = torch.ones(targets_without_padd.shape[0]).to(self.device)

            # word-level loss   (each prediction against each target)
            word_losses += self.criterion(
                preds_without_padd,
                targets_without_padd,
                y
            )

            y = torch.ones(1).to(self.device)

            image_embedding = images_embedding[i].unsqueeze(0)

            # 1º input loss (sentence predicted against input image)
            input1_losses += self.criterion(
                final_hs,
                image_embedding,
                y
            )

        word_loss = word_losses / n_sentences
        input1_loss = input1_losses / n_sentences

        loss = word_loss + input1_loss

        return loss
