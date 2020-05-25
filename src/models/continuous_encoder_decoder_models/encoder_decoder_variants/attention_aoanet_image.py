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
from preprocess_data.tokens import OOV_TOKEN
from embeddings.embeddings import EmbeddingsType
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_image import ContinuousAttentionImageModel
from embeddings.embeddings import EmbeddingsType
from preprocess_data.tokens import START_TOKEN, END_TOKEN
from models.abtract_model import DecodingType


class ContinuousDecoderSoftWithAoANet(DecoderWithAttention):
    """
    Decoder.
    """

    def __init__(
            self, attention_dim, embedding_type, embed_dim, decoder_dim, vocab_size, token_to_id, encoder_dim=2048,
            dropout=0.5):

        super(ContinuousDecoderSoftWithAoANet, self).__init__(attention_dim, embedding_type,
                                                              embed_dim, decoder_dim, vocab_size, token_to_id, encoder_dim, dropout)

        # linear layer to find representation of image
        self.represent_image = nn.Linear(encoder_dim, embed_dim)
        self.image_embedding = None

        self.aoa_layer = nn.Sequential(nn.Linear(decoder_dim + encoder_dim, encoder_dim*2), nn.GLU())

        # replace softmax layer with embedding layer
        self.fc = nn.Linear(encoder_dim, embed_dim)

    def init_hidden_state(self, mean_encoder_out):

        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        self.image_embedding = self.represent_image(mean_encoder_out)

        return h, h

    def forward(self, word, encoder_out, decoder_hidden_state, decoder_cell_state, mean_encoder_out, context_vector):
        embeddings = self.embedding(word)

        # TODO: RETURN self.mean_encoder_out
        lstm_input = torch.cat([embeddings, mean_encoder_out + context_vector], 1)

        decoder_hidden_state, decoder_cell_state = self.decode_step(
            lstm_input, (decoder_hidden_state, decoder_hidden_state))

        attention_weighted_encoding, alpha = self.attention(encoder_out, decoder_hidden_state)

        h_with_att = torch.cat([attention_weighted_encoding, decoder_hidden_state], 1)

        context_vector = self.aoa_layer(h_with_att)

        scores = self.fc(self.dropout(context_vector))

        return scores, decoder_hidden_state, decoder_cell_state, context_vector, alpha


class ContinuousAttentionAoANetImageModel(ContinuousAttentionImageModel):

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

        self.decoder = ContinuousDecoderSoftWithAoANet(
            encoder_dim=self.encoder.encoder_dim,
            attention_dim=self.args.attention_dim,
            decoder_dim=self.args.decoder_dim,
            embedding_type=self.args.embedding_type,
            embed_dim=self.args.embed_dim,
            vocab_size=self.vocab_size,
            token_to_id=self.token_to_id,
            dropout=self.args.dropout
        )

        self.decoder.normalize_embeddings()

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

    def _predict(self, encoder_out, caps, caption_lengths):
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)
        encoder_dim = encoder_out.size(2)

        # Create tensors to hold word predicion scores and alphas
        all_predictions = torch.zeros(batch_size,  max(
            caption_lengths), self.decoder.embed_dim).to(self.device)
        all_alphas = torch.zeros(batch_size, max(
            caption_lengths), num_pixels).to(self.device)

        mean_encoder_out = encoder_out.mean(dim=1)

        h, c = self.decoder.init_hidden_state(mean_encoder_out)
        context_vector = torch.zeros(batch_size, encoder_dim).to(self.device)

        # Predict
        for t in range(max(
                caption_lengths)):
            # batchsizes of current time_step are the ones with lenght bigger than time-step (i.e have not fineshed yet)
            batch_size_t = sum([l > t for l in caption_lengths])

            predictions, h, c, context_vector, alpha = self.decoder(
                caps[: batch_size_t, t],
                encoder_out[: batch_size_t],
                h[: batch_size_t],
                c[: batch_size_t],
                mean_encoder_out[: batch_size_t],
                context_vector[: batch_size_t]
            )

            all_predictions[:batch_size_t, t, :] = predictions
            all_alphas[:batch_size_t, t, :] = alpha

        return {"predictions": all_predictions, "alphas": all_alphas}

    def inference_with_greedy(self, image):
        with torch.no_grad():  # no need to track history

            decoder_sentence = START_TOKEN + " "

            input_word = torch.tensor([self.token_to_id[START_TOKEN]])

            i = 1

            encoder_output = self.encoder(image)
            encoder_output = encoder_output.view(
                1, -1, encoder_output.size()[-1])

            self.mean_encoder_out = encoder_output.mean(dim=1)
            h, c = self.decoder.init_hidden_state(self.mean_encoder_out)

            encoder_dim = encoder_output.size(2)
            self.context_vector = torch.zeros(1, encoder_dim).to(self.device)

            while True:

                scores, h, c = self.generate_output_index(
                    input_word, encoder_output, h, c)

                sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)

                current_output_index = sorted_indices[0]

                current_output_token = self.id_to_token[current_output_index.item(
                )]

                decoder_sentence += " " + current_output_token

                if (current_output_token == END_TOKEN or
                        i >= self.max_len-1):  # until 35
                    break

                input_word[0] = current_output_index.item()

                i += 1

            print("\ndecoded sentence", decoder_sentence)

            return decoder_sentence  # input_caption

    def inference_with_beamsearch(self, image, n_solutions=1):

        def compute_probability(seed_text, seed_prob, sorted_scores, index, current_text):
            return (seed_prob*len(seed_text) + np.log(sorted_scores[index].item()) / (len(seed_text)+1))

        def compute_perplexity(seed_text, seed_prob, sorted_scores, index, current_text):
            tokens = self.language_model_tokenizer.encode(current_text)
            input_ids = torch.tensor(tokens).unsqueeze(0)
            outputs = self.language_model(input_ids, labels=input_ids)
            loss, logits = outputs[:2]
            return math.exp(loss / len(tokens))

        def compute_sim2image(seed_text, seed_prob, sorted_scores, index, current_text):
            # sentence_mean = torch.zeros(1, embedding_dim)
            # for token in current_text:
            #     target_embedding(sentence_mean)

            # cosine_similarity(sentence_mean, self.image_repr)
            return 0

        def compute_perplexity_with_sim2image():
            return 0

        def generate_n_solutions(seed_text, seed_prob, encoder_out,  h, c,  n_solutions):
            last_token = seed_text[-1]

            if last_token == END_TOKEN:
                return [(seed_text, seed_prob, h, c)]

            top_solutions = []
            scores, h, c = self.generate_output_index(
                torch.tensor([self.token_to_id[last_token]]), encoder_out, h, c)

            sorted_scores, sorted_indices = torch.sort(
                scores, descending=True, dim=-1)

            for index in range(n_solutions):
                text = seed_text + \
                    [self.id_to_token[sorted_indices[index].item()]]
                # beam search taking into account lenght of sentence
                #prob = (seed_prob*len(seed_text) + np.log(sorted_scores[index].item()) / (len(seed_text)+1))
                text_score = compute_score(seed_text, seed_prob, sorted_scores, index, text)
                top_solutions.append((text, text_score, h, c))

            return top_solutions

        def get_most_probable(candidates, n_solutions):
            return sorted(candidates, key=operator.itemgetter(1), reverse=True)[:n_solutions]

        with torch.no_grad():
            encoder_output = self.encoder(image)
            encoder_output = encoder_output.view(
                1, -1, encoder_output.size()[-1])

            self.mean_encoder_out = encoder_output.mean(dim=1)
            h, c = self.decoder.init_hidden_state(self.mean_encoder_out)

            encoder_dim = encoder_output.size(2)
            self.context_vector = torch.zeros(1, encoder_dim).to(self.device)

            top_solutions = [([START_TOKEN], 0.0, h, c)]

            if self.args.decodying_type == DecodingType.BEAM.value:
                compute_score = compute_probability

            elif self.args.decodying_type == DecodingType.BEAM_PERPLEXITY.value:
                compute_score = compute_perplexity
                self.language_model_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
                self.language_model = GPT2LMHeadModel.from_pretrained('gpt2-xl')

            elif self.args.decodying_type == DecodingType.BEAM_SIM2IMAGE.value:
                compute_score = compute_sim2image

            elif self.args.decodying_type == DecodingType.BEAM_PERPLEXITY_SIM2IMAGE.value:
                compute_score = compute_perplexity_with_sim2image
                self.language_model_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
                self.language_model = GPT2LMHeadModel.from_pretrained('gpt2-xl')

            else:
                raise Exception("not available any other decoding type")

            for _ in range(self.max_len):
                candidates = []
                for sentence, prob, h, c in top_solutions:
                    candidates.extend(generate_n_solutions(
                        sentence, prob, encoder_output, h, c,  n_solutions))

                top_solutions = get_most_probable(candidates, n_solutions)

            # print("top solutions", [(text, prob)
            #                         for text, prob, _, _ in top_solutions])

            best_tokens, prob, h, c = top_solutions[0]

            best_sentence = " ".join(best_tokens)

            print("\nbeam decoded sentence:", best_sentence)
            return best_sentence

    def generate_output_index(self, input_word, encoder_out, h, c):
        predictions, h, c, context_vector, alpha = self.decoder(
            input_word, encoder_out, h, c, self.mean_encoder_out, self.context_vector)

        current_output_index = self._convert_prediction_to_output(predictions)

        self.context_vector = context_vector

        return current_output_index, h, c
