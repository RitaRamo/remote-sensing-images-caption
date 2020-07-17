import torchvision
from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from models.basic_encoder_decoder_models.encoder_decoder import Encoder, Decoder
import torch.nn.functional as F
from embeddings.embeddings import get_embedding_layer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from data_preprocessing.preprocess_tokens import OOV_TOKEN
from embeddings.embeddings import EmbeddingsType
from models.abtract_model import AbstractEncoderDecoderModel


class ContinuousDecoder(Decoder):
    """
    Decoder.
    """

    def __init__(self, decoder_dim, embed_dim, embedding_type, vocab_size, token_to_id, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(ContinuousDecoder, self).__init__(decoder_dim, embed_dim,
                                                embedding_type, vocab_size, token_to_id, encoder_dim, dropout)
        # instead of being bla bla
        self.fc = nn.Linear(decoder_dim, embed_dim)


class ContinuousMarginModel(AbstractEncoderDecoderModel):

    MODEL_DIRECTORY = "experiments/results/continuous_models/"

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
            embedding_type=EmbeddingsType.GLOVE.value,
            embed_dim=self.args.embed_dim,
            vocab_size=self.vocab_size,
            token_to_id=self.token_to_id,
            dropout=self.args.dropout
        )

        self.decoder.normalize_embeddings()

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

    def _define_loss_criteria(self):
        if self.args.loss == "cosine":
            self.apply_loss_type = self.cosine_embedding
            self.criterion = nn.CosineEmbeddingLoss().to(self.device)

        elif self.args.loss == "cosine":
            self.apply_loss_type = self.margin_loss
            self.criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    def _predict(self, encoder_out, caps, caption_lengths):
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

        # return is a dict, since for other models additional things maybe be return: ex attention model-> add alphas
        return {"predictions": all_predictions}

    def _calculate_loss(self, predict_output, caps, caption_lengths):
        predictions = predict_output["predictions"]
        targets = caps[:, 1:]  # targets doesnt have stark token
        print("this is targets", targets)

        predictions = pack_padded_sequence(
            predictions, caption_lengths, batch_first=True)
        targets = pack_padded_sequence(
            targets, caption_lengths, batch_first=True)

        # print("is equal", torch.equal(self.decoder.embedding.weight.data[targets.data[0]].unsqueeze_(0), self.decoder.embedding(
        #     torch.tensor(targets.data[0])).unsqueeze_(0)))

        # prim_embedd = self.decoder.embedding.weight.data[targets.data[0]].unsqueeze_(
        #     0)

        # prim_with_deco = self.decoder.embedding(
        #     torch.tensor(targets.data[0])).unsqueeze_(0)

        # print("shape of prim embe", prim_embedd.size())
        # print("shape of prim embe with dec", prim_with_deco.size())
        # print("shape of WITHout SEQUEZE embe",
        #       self.decoder.embedding.weight.data[targets.data[0]].size())

        # target_embeddings = self.decoder.embedding(
        #     torch.tensor(targets.data[0])).unsqueeze_(0)

        # for index_word in targets.data[1:]:
        #     embedding_target = self.decoder.embedding(
        #         torch.tensor(index_word)).unsqueeze_(0)
        #     target_embeddings = torch.cat(
        #         (target_embeddings, embedding_target), 0)

        target_embeddings = torch.zeros(
            predictions.data.size()[0], predictions.data.size()[1])

        for i in range(targets.data.size()[0]):
            index_word = targets.data[i]
            print("index word", index_word)
            embedding_target = self.decoder.embedding(index_word)
            target_embeddings[i, :] = embedding_target

        print("this is target embeddings", target_embeddings)

        target_embeddings = target_embeddings.to(self.device)
        y = torch.ones(target_embeddings.shape[0]).to(self.device)

        loss = self.apply_loss_type(predictions.data, target_embeddings)

        # negative_examples = torch.zeros(target_embeddings.size()[
        #     0], target_embeddings.size()[1])

        # print("shape of negative examples", negative_examples.size())

        # for i in range(len(target_embeddings)):

        #     output_similarity_to_embeddings = cosine_similarity(target_embeddings[i].unsqueeze_(0),
        #                                                         self.decoder.embedding.weight.data)[0]

        #     id_of_most_similar_embedding_except_itself = np.argsort(
        #         output_similarity_to_embeddings)[::-1][1]

        #     nearest_neighbour_embedding = self.decoder.embedding.weight.data[
        #         id_of_most_similar_embedding_except_itself]

        #     negative_examples[i, :] = nearest_neighbour_embedding

        #     # print("this is target i", targets.data[i])

        #     print("this is target_embeddings i size",
        #           target_embeddings[i].unsqueeze_(0).size())

        #     # print("closeste argmax", np.argmax(
        #     #     output_similarity_to_embeddings))

        #     # print("this is arg sort first element", np.argsort(
        #     #     output_similarity_to_embeddings)[0])

        #     # print("\ntop 5 most similar with argsort", np.argsort(
        #     #     output_similarity_to_embeddings)[::-1][:5])

        #     # print("the second most similar is",
        #     #       id_of_most_similar_embedding_except_itself)

        #     # print("this is the word of target",
        #     #       self.id_to_token[targets.data[i].item()])
        #     # print("this represents the word",
        #     #       self.id_to_token[id_of_most_similar_embedding_except_itself])

        #     #print("this is nearest_neighbour", nearest_neighbour)

        #     # break

        # print("negative examples", negative_examples)

        # var1 = self.my_first_method
        # var1("RITA")

        # loss = self.criterion(
        #     predictions.data, target_embeddings, negative_examples)

        return loss

    def margin_loss(self, predictions, target_embeddings, pre):

        negative_examples = torch.zeros(target_embeddings.size()[
                                        0], target_embeddings.size()[1])

        for i in range(len(target_embeddings)):

            output_similarity_to_embeddings = cosine_similarity(target_embeddings[i].unsqueeze_(0),
                                                                self.decoder.embedding.weight.data)[0]

            id_of_most_similar_embedding_except_itself = np.argsort(
                output_similarity_to_embeddings)[::-1][1]

            nearest_neighbour_embedding = self.decoder.embedding.weight.data[
                id_of_most_similar_embedding_except_itself]

            negative_examples[i, :] = nearest_neighbour_embedding

        #print("negative examples", negative_examples)

        loss = self.criterion(
            predictions, target_embeddings, negative_examples)

        return loss

    def my_first_method(self, name):
        print("AHHHHHHHH" + name)
        return 0

    # def apply_loss(method_loss):
        # return metdo_loss
    # self.loss()

    # def margin_loss():

    # def cosine:
        # y=torch
        # return criterion()...

    def generate_output_index(self, input_word, encoder_out, h, c):
        predictions, h, c = self.decoder(
            input_word, encoder_out, h, c)

        current_output_index = self._convert_prediction_to_output(predictions)

        return current_output_index, h, c

    def _convert_prediction_to_output(self, predictions):
        output_similarity_to_embeddings = cosine_similarity(
            self.decoder.embedding.weight.data, predictions)

        current_output_index = np.argmax(output_similarity_to_embeddings)

        return current_output_index
