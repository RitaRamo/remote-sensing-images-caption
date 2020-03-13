import torchvision
from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from models.basic_model import BasicModel, Encoder, Decoder
import torch.nn.functional as F
from embeddings.embeddings import get_embedding_layer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from preprocess_data.tokens import OOV_TOKEN
from embeddings.embeddings import EmbeddingsType


class ContinuousDecoder(Decoder):
    """
    Decoder.
    """

    def __init__(self, decoder_dim,  embed_dim, embedding_type, vocab_size, token_to_id, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(ContinuousDecoder, self).__init__(decoder_dim,  embed_dim,
                                                embedding_type, vocab_size, token_to_id, encoder_dim, dropout)
        # instead of being bla bla
        self.fc = nn.Linear(decoder_dim, embed_dim)


class BasicContinuousModel(BasicModel):

    MODEL_DIRECTORY = "experiments/results/continuous_model/"

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

        self.encoder = Encoder(self.args.image_model_type,
                               enable_fine_tuning=self.args.fine_tune_encoder)

        self.decoder = ContinuousDecoder(
            encoder_dim=self.encoder.encoder_dim,
            decoder_dim=self.args.decoder_dim,
            embedding_type=EmbeddingsType.GLOVE_FOR_CONTINUOUS_MODELS.value,
            embed_dim=self.args.embed_dim,
            vocab_size=self.vocab_size,
            token_to_id=self.token_to_id,
            dropout=self.args.dropout
        )

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

    def _define_loss_criteria(self):
        self.criterion = nn.CosineEmbeddingLoss().to(self.device)

    def _calculate_loss(self, encoder_out, caps, caption_lengths):
        batch_size = encoder_out.size(0)

        # Create tensors to hold word predicion scores
        all_predictions = torch.zeros(batch_size, max(
            caption_lengths), self.decoder.embed_dim).to(self.device)

        h, c = self.decoder.init_hidden_state(encoder_out)

        # Predict
        for t in range(max(caption_lengths)):
            # batchsizes of current time_step are the ones with lenght bigger than time-step (i.e have not fineshed yet)
            batch_size_t = sum([l > t for l in caption_lengths])

            predictions, h, c = self.decoder(
                caps[:batch_size_t, t], encoder_out[:batch_size_t], h[:batch_size_t], c[:batch_size_t])

            all_predictions[:batch_size_t, t, :] = predictions

        # Calculate loss
        targets = caps[:, 1:]  # targets doesnt have stark token

        scores = pack_padded_sequence(
            all_predictions, caption_lengths, batch_first=True)
        targets = pack_padded_sequence(
            targets, caption_lengths, batch_first=True)

        target_embeddings = self.decoder.embedding(
            torch.tensor(targets.data[0])).unsqueeze_(0)

        for index_word in targets.data[1:]:
            embedding_target = self.decoder.embedding(
                torch.tensor(index_word)).unsqueeze_(0)
            target_embeddings = torch.cat(
                (target_embeddings, embedding_target), 0)

        target_embeddings = target_embeddings.to(self.device)
        y = torch.ones(target_embeddings.shape[0]).to(self.device)

        loss = self.criterion(scores.data, target_embeddings, y)

        return loss

    def generate_output_index(self, input_word, encoder_out, h, c):
        predicted_embedding_output, h, c = self.decoder(
            input_word, encoder_out, h, c)

        output_similarity_to_embeddings = cosine_similarity(
            self.decoder.embedding.weight.data, predicted_embedding_output)

        current_output_index = np.argmax(output_similarity_to_embeddings)

        return current_output_index, h, c
