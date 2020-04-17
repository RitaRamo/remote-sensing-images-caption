import torchvision
from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from models.basic_encoder_decoder_models.encoder_decoder_variants.attention import Attention, Encoder
from models.abtract_model import AbstractEncoderDecoderModel
import torch.nn.functional as F
from embeddings.embeddings import get_embedding_layer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from preprocess_data.tokens import OOV_TOKEN, END_TOKEN, START_TOKEN
from embeddings.embeddings import EmbeddingsType
from models.continuous_encoder_decoder_models.encoder_decoder import ContinuousEncoderDecoderModel
from transformers import BertModel, BertTokenizer


class BertDecoder(nn.Module):

    def __init__(
            self, bert_tokenizer, bert_model, decoder_dim, vocab_size, token_to_id, id_to_token, encoder_dim=2048,
            dropout=0.5):

        super(BertDecoder, self).__init__()
        self.encoder_dim = encoder_dim

        self.embed_dim = 768  # BERT embedding dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.id_to_token = id_to_token
        self.dropout = dropout

        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(
            self.embed_dim, decoder_dim, bias=True)  # decoding LSTMCell
        # linear layer to find initial hidden state of LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        # linear layer to find initial cell state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        self.fc = nn.Linear(decoder_dim, self.embed_dim)
        self.init_weights()  # initialize some layers with the uniform distribution

        self.bert_tokenizer = bert_tokenizer
        self.bert_model = bert_model

        self.bert_matrix_embedding = torch.load("bert_matrix.pth.tar")[
            "pretrained_embeddings_matrix"]

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

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

    def get_contextualize_embedding(self, current_word_and_previous_ones):
        input_ids = self.bert_tokenizer.encode(
            current_word_and_previous_ones, add_special_tokens=False)
        # Mark each of the inputs belonging to sentence "1".
        segments_ids = [1] * len(input_ids)

        tokens_tensor = torch.tensor([input_ids])
        segments_tensors = torch.tensor([segments_ids])

        with torch.no_grad():
            last_hidden_states = self.bert_model(
                tokens_tensor, segments_tensors)[0].squeeze(dim=0)  # [1,x]; [#1, x, ]

        current_word = current_word_and_previous_ones.split()[-1]
        # in case current word is OOV and is replaced by x subwords, having x embeddings
        n_bert_tokens_for_current_word = len(
            self.bert_tokenizer.tokenize(current_word))

        embeddings_of_current_word = last_hidden_states[-
                                                        n_bert_tokens_for_current_word:, :]

        embedding_of_current_word = torch.mean(
            embeddings_of_current_word, dim=0)

        return embedding_of_current_word

    def get_inputs_contextualized_embeddings(self, batch_words):

        batch_size = batch_words.size(0)
        batch_embeddings = torch.zeros(batch_size, self.embed_dim)
        # two_in=torch.tensor([tokenizer.encode("diff is some text to lixo", add_special_tokens=True), tokenizer.encode("diff is some text lixo to", add_special_tokens=True)])

        for i in range(batch_size):
            caption = list(batch_words[i, :])

            # replace start token with start token of bert
            caption[0] = "[CLS]"

            current_word_and_previous_ones = " ".join(
                [caption[0]] +
                [self.id_to_token[id_word.item()] for id_word in caption[1:]]
            )

            batch_embeddings[i, :] = self.get_contextualize_embedding(
                current_word_and_previous_ones)

        return batch_embeddings

    def forward(self, batch_words, encoder_out, decoder_hidden_state, decoder_cell_state):

        embeddings = self.get_inputs_contextualized_embeddings(
            batch_words)

        decoder_hidden_state, decoder_cell_state = self.decode_step(
            embeddings, (decoder_hidden_state, decoder_cell_state)
        )

        scores = self.fc(self.dropout(decoder_hidden_state))

        return scores, decoder_hidden_state, decoder_cell_state


class ContinuousBertModel(ContinuousEncoderDecoderModel):

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
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.eval()

        self.encoder = Encoder(self.args.image_model_type,
                               enable_fine_tuning=self.args.fine_tune_encoder)

        self.decoder = BertDecoder(
            bert_tokenizer=self.bert_tokenizer,
            bert_model=self.bert_model,
            encoder_dim=self.encoder.encoder_dim,
            decoder_dim=self.args.decoder_dim,
            vocab_size=self.vocab_size,
            token_to_id=self.token_to_id,
            id_to_token=self.id_to_token,
            dropout=self.args.dropout
        )

        # self.decoder.normalize_embeddings()

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

    def _predict(self, encoder_out, caps, caption_lengths):
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)

        # Create tensors to hold word predicion scores and alphas
        all_predictions = torch.zeros(batch_size,  max(
            caption_lengths), self.decoder.embed_dim).to(self.device)
        all_alphas = torch.zeros(batch_size, max(
            caption_lengths), num_pixels).to(self.device)

        h, c = self.decoder.init_hidden_state(encoder_out)

        # Predict
        for t in range(max(caption_lengths)):
            # batchsizes of current time_step are the ones with lenght bigger than time-step (i.e have not fineshed yet)
            batch_size_t = sum([l > t for l in caption_lengths])

            # caps[:batch_size_t, :t] instead of caps[:batch_size_t, t] to have current word with previous words
            predictions, h, c = self.decoder(
                caps[:batch_size_t, 0:t+1], encoder_out[:batch_size_t], h[:batch_size_t], c[:batch_size_t])
            # not inclusive, hence t+1
            all_predictions[:batch_size_t, t, :] = predictions

        return {"predictions": all_predictions, "alphas": all_alphas}

    def _calculate_loss(self, predict_output, caps, caption_lengths):
        predictions = predict_output["predictions"]
        targets = caps[:, 1:]  # targets doesnt have stark token

        batch_size = predictions.size()[0]
        n_tokens = predictions.size()[1]
        embeddings_dim = predictions.size()[-1]

        targets_embeddings = torch.zeros(
            batch_size,
            n_tokens,
            embeddings_dim
        )

        for i in range(batch_size):
            # do not reach end_token (we will need to replace by SEP token)
            current_word_and_previous_ones = ""
            for t in range(caption_lengths[i]-1):

                id_word = targets[i, t]  # word of caption_i of time-step_t

                current_word_and_previous_ones += self.id_to_token[id_word.item(
                )] + " "

                embedding_target = self.decoder.get_contextualize_embedding(
                    current_word_and_previous_ones)

                targets_embeddings[i, t, :] = embedding_target

            # last embedding: end_token replaced for SEP token
            current_word_and_previous_ones += "[SEP]"
            embedding_target = self.decoder.get_contextualize_embedding(
                current_word_and_previous_ones)
            targets_embeddings[i, t+1, :] = embedding_target

        predictions = pack_padded_sequence(
            predictions, caption_lengths, batch_first=True)
        targets_embeddings = pack_padded_sequence(
            targets_embeddings, caption_lengths, batch_first=True)

        targets_embeddings = targets_embeddings.to(self.device)

        loss = self.criteria.compute_loss(
            predictions.data, targets_embeddings.data, self.bert_matrix_embedding)

        return loss

    def generate_text(self, image):
        with torch.no_grad():  # no need to track history

            decoder_sentence = START_TOKEN + " "

            input_words_index = torch.zeros(
                (1, self.max_len-1), dtype=torch.int32)

            input_words_index[0] = self.token_to_id[START_TOKEN]

            i = 1

            encoder_output = self.encoder(image)
            encoder_output = encoder_output.view(
                1, -1, encoder_output.size()[-1])

            h, c = self.decoder.init_hidden_state(encoder_output)

            while True:
                current_output_index, h, c = self.generate_output_index(
                    input_words_index[:, :i], encoder_output, h, c)

                current_output_token = self.id_to_token[current_output_index.item(
                )]

                decoder_sentence += " " + current_output_token

                if (current_output_token == END_TOKEN or
                        i >= self.max_len-1):  # until 35
                    break

                input_words_index[:, i] = current_output_index.item()

                i += 1

            print("\ndecoded sentence", decoder_sentence)

            return decoder_sentence  # input_caption

    def generate_output_index(self, input_words_index, encoder_out, h, c):
        predictions, h, c = self.decoder(
            input_words_index, encoder_out, h, c)

        current_output_index = self._convert_prediction_to_output(input_words_index, predictions)

        return current_output_index, h, c

    def _convert_prediction_to_output(self, input_words_index, predictions):

        predictions = predictions.repeat(self.vocab_size, 1)

        output_similarity_to_embeddings = torch.cosine_similarity(self.decoder.bert_matrix_embedding, predictions)

        top_scores, top_indices = torch.topk(output_similarity_to_embeddings, k=1, dim=0)

        current_output_index = top_indices[0]

        return current_output_index
