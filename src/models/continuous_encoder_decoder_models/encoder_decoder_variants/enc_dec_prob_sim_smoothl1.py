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
from data_preprocessing.preprocess_tokens import START_TOKEN, END_TOKEN, PAD_TOKEN


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


class ContinuousEncoderDecoderProbSimSmoothl1Model(ContinuousEncoderDecoderModel):

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

    def _define_loss_criteria(self):
        self.criterion_ce = nn.CrossEntropyLoss().to(self.device)
        self.criterion_sim = nn.SmoothL1Loss(reduction='none').to(self.device)

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

        predictions = pack_padded_sequence(
            predictions, caption_lengths, batch_first=True)
        targets = pack_padded_sequence(
            targets, caption_lengths, batch_first=True)

        loss_ce = self.criterion_ce(predictions.data, targets.data)

        predictions_embeddings = predict_output["predictions2"]

        if self.args.no_normalization == False:
            # when target embeddings start normalized, predictions should also be normalized
            predictions_embeddings = torch.nn.functional.normalize(predictions_embeddings, p=2, dim=-1)

        predictions_embeddings = pack_padded_sequence(
            predictions_embeddings, caption_lengths, batch_first=True)
        target_embeddings = pack_padded_sequence(
            target_embeddings, caption_lengths, batch_first=True)

        loss_sim = self.criterion_sim(predictions_embeddings.data, target_embeddings.data)
        loss_sim = torch.sum(loss_sim, dim=-1)
        loss_sim = torch.mean(loss_sim)

        loss = loss_ce + loss_sim

        return loss

    def generate_output_index(self, input_word, encoder_out, h, c):
        predictions1, predictions2, h, c = self.decoder(
            input_word, encoder_out, h, c)

        current_output_index = self._convert_prediction_to_output(predictions1)

        return current_output_index, h, c

    def _convert_prediction_to_output(self, predictions):
        scores = F.log_softmax(predictions, dim=1)  # more stable
        # scores = F.softmax(predictions, dim=1)[0]  # actually probs
        return scores

    def inference_with_greedy_and_sim_rank(
            self, image, n_solutions=0, min_len=0, repetition_window=0, max_len=50):
        with torch.no_grad():  # no need to track history

            decoder_sentence = []

            input_word = torch.tensor([self.token_to_id[START_TOKEN]])

            i = 1

            encoder_output = self.encoder(image)
            encoder_output = encoder_output.view(
                1, -1, encoder_output.size()[-1])

            h, c = self.decoder.init_hidden_state(encoder_output)

            while True:

                predictions1, predictions2, h, c = self.decoder(input_word, encoder_output, h, c)

                scores = F.log_softmax(predictions1, dim=1)

                sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)

                most_sim_index = 0
                highest_sim_score = 0
                for top_index in sorted_indices.squeeze()[:n_solutions]:
                    embedding_argmax = self.decoder.embedding(top_index)

                    sim_argmax_embedding_vs_predicted_embedding = torch.cosine_similarity(
                        embedding_argmax, predictions2, dim=-1)

                    if sim_argmax_embedding_vs_predicted_embedding > highest_sim_score:
                        most_sim_index = top_index
                        highest_sim_score = sim_argmax_embedding_vs_predicted_embedding

                current_output_index = most_sim_index

                current_output_token = self.id_to_token[current_output_index.item()]

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
