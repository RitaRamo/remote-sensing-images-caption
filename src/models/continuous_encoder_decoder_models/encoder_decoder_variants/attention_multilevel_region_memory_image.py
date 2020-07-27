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


class MultilevelAttention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, c_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(MultilevelAttention, self).__init__()
        # linear layer to transform encoded image
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.c_att = nn.Linear(c_dim, attention_dim)
        self.decoder_encoder_att = nn.Linear(decoder_dim, attention_dim)
        self.decoder_c_att = nn.Linear(decoder_dim, attention_dim)

        self.attention1_att = nn.Linear(attention_dim, attention_dim)
        self.attention2_att = nn.Linear(attention_dim, attention_dim)
        self.decoder_attention1_att = nn.Linear(decoder_dim, attention_dim)
        self.decoder_attention2_att = nn.Linear(decoder_dim, attention_dim)

        # linear layer to calculate values to be softmax-ed
        self.full_att = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_features, cell_states, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        w_i = self.encoder_att(encoder_features)  # (batch_size, n_regions, attention_dim)
        w_h = self.decoder_encoder_att(decoder_hidden).unsqueeze(1)  # (batch_size, 1, attention_dim)
        q_att = self.full_att(self.tanh(w_i + w_h)).squeeze(2)  # (batch_size, n_regions) (with squeeze)
        alpha_i = self.softmax(q_att)  # (batch_size, n_regions)
        # (batch_size, regions_dim)
        attention1 = (w_i * alpha_i.unsqueeze(2)).sum(dim=1)

        w_c = self.c_att(cell_states)  # (batch_size, 1, attention_dim)
        w_h = self.decoder_c_att(decoder_hidden).unsqueeze(1)  # (batch_size, 1, attention_dim)
        s_att = self.full_att(self.tanh(w_c + w_h)).squeeze(2)
        alpha_s = self.softmax(s_att)  # (batch_size, 1)
        attention2 = (cell_states * alpha_s.unsqueeze(2)).sum(dim=1)  # (batch_size, c_dim)

        w_attention1 = self.attention1_att(attention1).unsqueeze(1)
        w_h = self.decoder_attention1_att(decoder_hidden).unsqueeze(1)  # (batch_size, 1, attention_dim)
        att_att1 = self.full_att(self.tanh(w_attention1 + w_h)).squeeze(2)
        alpha_att1 = self.softmax(att_att1)

        w_attention2 = self.attention2_att(attention2).unsqueeze(1)
        w_h = self.decoder_attention2_att(decoder_hidden).unsqueeze(1)  # (batch_size, 1, attention_dim)
        att_att2 = self.full_att(self.tanh(w_attention2 + w_h)).squeeze(2)
        alpha_att2 = self.softmax(att_att2)

        attention_weighted_encoding = alpha_att1 * attention1 + alpha_att2 * attention2

        return attention_weighted_encoding, alpha_i, alpha_s, alpha_att1, alpha_att2


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
        self.attention = MultilevelAttention(
            encoder_dim, decoder_dim, decoder_dim, attention_dim)  # attention network

        # replace softmax layer with embedding layer
        self.decode_step = nn.LSTMCell(embed_dim + attention_dim, decoder_dim, bias=True)
        self.fc = nn.Linear(decoder_dim + attention_dim, embed_dim)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)

        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim) 512 units
        c = self.init_c(mean_encoder_out)  # (batch_size, decoder_dim) 512 units

        self.image_embedding = self.represent_image(mean_encoder_out)  # 300 512

        z_context, _, _, _, _ = self.attention(encoder_out, c, h)

        return h, c, z_context

    def forward(
            self, word, encoder_out, old_decoder_hidden_state, old_decoder_cell_state, old_decoder_cell_states,
            old_z_context):
        embeddings = self.embedding(word)
        decoder_input = torch.cat((embeddings, old_z_context), dim=1)

        # current decoder hidden state is the one of last time step
        decoder_hidden_state, decoder_cell_state = self.decode_step(
            decoder_input, (old_decoder_hidden_state, old_decoder_cell_state)
        )

        attended_decoder_cells = torch.cat((old_decoder_cell_states, decoder_cell_state.unsqueeze(1)), dim=1)

        z_context, alpha_i, alpha_s, alpha_att1, alpha_att2 = self.attention(
            encoder_out, attended_decoder_cells, decoder_hidden_state)

        scores = self.fc(self.dropout(torch.cat((decoder_hidden_state, z_context), dim=1)))

        return scores, decoder_hidden_state, decoder_cell_state, z_context


class ContinuousAttentionMultilevelRegionMemoryImageModel(ContinuousAttentionModel):

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

    def _predict(self, encoder_out, caps, caption_lengths):
        torch.autograd.set_detect_anomaly(True)
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)

        # Create tensors to hold word predicion scores and alphas
        all_predictions = torch.zeros(batch_size, max(
            caption_lengths), self.decoder.embed_dim).to(self.device)

        h, c, z_context = self.decoder.init_hidden_state(encoder_out)

        # +1,since cell state needs to start with 1º c0 in pos0, contrary to predictions
        all_cs = torch.zeros(batch_size, max(caption_lengths) + 1, c.size(1)).to(self.device)
        all_cs[:, 0, :] = c

        # Predict
        for t in range(max(
                caption_lengths)):

            # batchsizes of current time_step are the ones with lenght bigger than time-step (i.e have not fineshed yet)
            batch_size_t = sum([l > t for l in caption_lengths])

            predictions, h, c, z_context = self.decoder(
                caps[: batch_size_t, t],
                encoder_out[: batch_size_t],
                h[: batch_size_t],
                c[: batch_size_t],
                all_cs[: batch_size_t, : t + 1],
                z_context[: batch_size_t])

            all_predictions[:batch_size_t, t, :] = predictions
            all_cs[:batch_size_t, t + 1, :] = c

        return {"predictions": all_predictions}

    def generate_output_embedding(self, input_embedding, encoder_out, h, c):
        predictions, h, c, _ = self.decoder.inference(
            input_embedding, encoder_out, h, c)

        current_output_index = self._convert_prediction_to_output(predictions)

        return predictions, current_output_index, h, c

    def inference_with_greedy(self, image, n_solutions=0):
        with torch.no_grad():  # no need to track history

            decoder_sentence = []

            input_word = torch.tensor([self.token_to_id[START_TOKEN]])

            i = 1

            encoder_output = self.encoder(image)
            encoder_output = encoder_output.view(
                1, -1, encoder_output.size()[-1])

            h, c, z_context = self.decoder.init_hidden_state(encoder_output)
            all_cs = torch.zeros(1, self.max_len, c.size(1)).to(self.device)
            all_cs[0, 0, :] = c

            while True:

                predictions, h, c, z_context = self.decoder(input_word, encoder_output, h, c, all_cs[:, :i], z_context)
                all_cs[0, i, :] = c
                scores = self._convert_prediction_to_output(predictions)

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

            return generated_sentence  # input_caption

    # def inference_with_beamsearch(self, image, n_solutions=3):

    #     def compute_probability(seed_text, seed_prob, sorted_scores, index, current_text):
    #         return (seed_prob * len(seed_text) + np.log(sorted_scores[index].item())) / (len(seed_text) + 1)

    #     def compute_perplexity(seed_text, seed_prob, sorted_scores, index, current_text):
    #         current_text = ' '.join(current_text)
    #         tokens = self.language_model_tokenizer.encode(current_text)

    #         input_ids = torch.tensor(tokens).unsqueeze(0)
    #         with torch.no_grad():
    #             outputs = self.language_model(input_ids, labels=input_ids)
    #             loss, logits = outputs[:2]

    #         return math.exp(loss / len(tokens))

    #     def compute_sim2image(seed_text, seed_prob, sorted_scores, index, current_text):
    #         n_tokens = len(current_text)
    #         tokens_ids = torch.zeros(1, n_tokens)
    #         for i in range(n_tokens):
    #             token = current_text[i]
    #             tokens_ids[0, i] = self.token_to_id[token]

    #         tokens_embeddings = self.decoder.embedding(tokens_ids.long()).to(self.device)

    #         sentence_mean = torch.mean(tokens_embeddings, dim=1)
    #         images_embedding = self.decoder.image_embedding

    #         return torch.cosine_similarity(sentence_mean, images_embedding)

    #     def compute_perplexity_with_sim2image():
    #         return 0

    #     def generate_n_solutions(seed_text, seed_prob, encoder_out, h, c,all_cs, time_t, n_solutions):
    #         last_token = seed_text[-1]

    #         if last_token == END_TOKEN:
    #             return [(seed_text, seed_prob, h, c)]

    #         top_solutions = []

    #         predictions, h, c, z_context = self.decoder( torch.tensor([self.token_to_id[last_token]]), encoder_out, h, c, all_cs[:,time_t], z_context)
    #         all_cs[0, i, :] = c
    #         scores = self._convert_prediction_to_output(predictions)

    #         sorted_scores, sorted_indices = torch.sort(
    #             scores, descending=True, dim=-1)

    #         for index in range(n_solutions):
    #             text = seed_text + [self.id_to_token[sorted_indices[index].item()]]
    #             # beam search taking into account lenght of sentence
    #             # prob = (seed_prob*len(seed_text) + np.log(sorted_scores[index].item()) / (len(seed_text)+1))
    #             text_score = compute_score(seed_text, seed_prob, sorted_scores, index, text)
    #             top_solutions.append((text, text_score, h, c))

    #         return top_solutions

    #     def get_most_probable(candidates, n_solutions, is_to_reverse):
    #         return sorted(candidates, key=operator.itemgetter(1), reverse=is_to_reverse)[:n_solutions]

    #     with torch.no_grad():
    #         encoder_output = self.encoder(image)
    #         encoder_output = encoder_output.view(1, -1, encoder_output.size()[-1])  # flatten encoder
    #         h, c, z_context = self.decoder.init_hidden_state(encoder_output)
    #         all_cs = torch.zeros(1, self.max_len, c.size(1)).to(self.device)
    #         all_cs[0, 0, :] = c

    #         top_solutions = [([START_TOKEN], 0.0, h, c, all_cs)]

    #         if self.args.decodying_type == DecodingType.BEAM.value:
    #             compute_score = compute_probability
    #             is_to_reverse = True

    #         elif self.args.decodying_type == DecodingType.BEAM_PERPLEXITY.value:
    #             compute_score = compute_perplexity
    #             is_to_reverse = False

    #         elif self.args.decodying_type == DecodingType.BEAM_SIM2IMAGE.value:
    #             compute_score = compute_sim2image

    #         elif self.args.decodying_type == DecodingType.BEAM_PERPLEXITY_SIM2IMAGE.value:
    #             compute_score = compute_perplexity_with_sim2image

    #         else:
    #             raise Exception("not available any other decoding type")

    #         for i in range(self.max_len):
    #             candidates = []
    #             for sentence, prob, h, c, all_cs in top_solutions:
    #                 candidates.extend(generate_n_solutions(
    #                     sentence, prob, encoder_output, h, c, all_cs, i, n_solutions))

    #             top_solutions = get_most_probable(candidates, n_solutions, is_to_reverse)

    #         # print("top solutions", [(text, prob)
    #         #                         for text, prob, _, _ in top_solutions])
    #         best_tokens, prob, h, c = top_solutions[0]

    #         if best_tokens[0] == START_TOKEN:
    #             best_tokens = best_tokens[1:]
    #         if best_tokens[-1] == END_TOKEN:
    #             best_tokens = best_tokens[:-1]
    #         best_sentence = " ".join(best_tokens)

    #         print("\nbeam decoded sentence:", best_sentence)
    #         return best_sentence
