import torchvision
from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from models.basic_encoder_decoder_models.encoder_decoder_variants.attention_scale_product import ScaleProductAttention, DecoderWithAttention
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
        self.init_h = nn.Linear(embed_dim, decoder_dim)


    def init_hidden_state(self, encoder_attr):

        h = self.init_h(encoder_attr)  # (batch_size, decoder_dim) 512 units

        return h, h



class ContinuousScaleProductAttentionHModel(ContinuousEncoderDecoderModel):

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

        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)

        # Create tensors to hold word predicion scores and alphas
        all_predictions = torch.zeros(batch_size, max(
            caption_lengths), self.decoder.embed_dim).to(self.device)
        all_alphas = torch.zeros(batch_size, max(
            caption_lengths), num_pixels).to(self.device)

        h, c = self.decoder.init_hidden_state(encoder_attrs)

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

    def inference_with_greedy_smoothl1(self, image, n_solutions=0, min_len=0, repetition_window=0, max_len=50):
        with torch.no_grad():  # no need to track history

            decoder_sentence = []

            input_word = torch.tensor([self.token_to_id[START_TOKEN]])

            i = 1

            encoder_output, encoder_attrs = self.encoder(image)

            encoder_output = encoder_output.view(
                1, -1, encoder_output.size()[-1])

            h, c = self.decoder.init_hidden_state(encoder_attrs)

            criteria = torch.nn.SmoothL1Loss(reduction="none")

            while True:

                scores, h, c = self.generate_output_index_smoothl1(criteria,
                                                                   input_word, encoder_output, h, c)

                sorted_scores, sorted_indices = torch.sort(scores, descending=False, dim=-1)
                # print("this are the sorted_scores", sorted_scores)
                # print("this are the sorted_indices", sorted_indices)
                # k_l = 0
                # for indi in sorted_indices:
                #     print(self.id_to_token[indi.item()], sorted_scores[k_l])
                #     k_l += 1
                #     if k_l > 5:
                #         break

                current_output_index = sorted_indices.squeeze()[0]
                # print("current output index", current_output_index)
                # if current_output_index.item() == self.token_to_id[PAD_TOKEN]:
                #     current_output_index = sorted_indices.squeeze()[1]

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
            # print("beam_t decoded sentence:", generated_sentence)
            print("\ngenerated sentence:", generated_sentence)

            return generated_sentence  # input_caption

    def generate_output_index_smoothl1(self, criteria, input_word, encoder_out, h, c):
        predictions, h, c,_ = self.decoder(
            input_word, encoder_out, h, c)

        current_output_index = self._convert_prediction_to_output_smoothl1(criteria, predictions)

        return current_output_index, h, c   
