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
import operator

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


class ContinuousScaleProductAttentionModel(ContinuousEncoderDecoderModel):

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
        self.decodying_criteria = torch.nn.SmoothL1Loss(reduction="none")

    def _predict(self, encoder_out, caps, caption_lengths):
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

    def generate_output_index_smoothl1(self, criteria, input_word, encoder_out, h, c):
        predictions, h, c,_ = self.decoder(
            input_word, encoder_out, h, c)

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


            best_tokens, prob, h, c = top_solutions[0]

            if best_tokens[0] == START_TOKEN:
                best_tokens = best_tokens[1:]
            if best_tokens[-1] == END_TOKEN:
                best_tokens = best_tokens[:-1]
            best_sentence = " ".join(best_tokens)

            print("\nbeam decoded sentence:", best_sentence)
            return best_sentence

    def inference_with_beamsearch(
            self, image, n_solutions=3, min_len=2, repetition_window=0, max_len=50):

        def compute_probability(seed_text, seed_prob, sorted_scores, index, current_text):
            # print("\nseed text", seed_text)
            # print("current_text text", current_text)
            # print("previous seed prob", seed_prob)
            # print("now prob", sorted_scores[index].item())
            # print("final prob", seed_prob + sorted_scores[index])
            # print("final prob with item", seed_prob + sorted_scores[index].item())

            # print(stop)
            return (seed_prob * len(seed_text) + sorted_scores[index].item()) / (len(seed_text) + 1)  # .item()

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


            best_tokens, prob, h, c = top_solutions[0]

            if best_tokens[0] == START_TOKEN:
                best_tokens = best_tokens[1:]
            if best_tokens[-1] == END_TOKEN:
                best_tokens = best_tokens[:-1]
            best_sentence = " ".join(best_tokens)

            print("\nbeam decoded sentence:", best_sentence)
            return best_sentence

    def generate_output_index(self, input_word, encoder_out, h, c):
        predictions, h, c,_ = self.decoder(
            input_word, encoder_out, h, c)

        current_output_index = self._convert_prediction_to_output(predictions)

        return current_output_index, h, c

