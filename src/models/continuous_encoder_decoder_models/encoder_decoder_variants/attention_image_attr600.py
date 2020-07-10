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
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention import ContinuousAttentionModel
from embeddings.embeddings import EmbeddingsType
from preprocess_data.tokens import START_TOKEN, END_TOKEN
from preprocess_data.images import ImageNetModelsPretrained
import logging
from torchvision import transforms, models


def get_image_extractor(model_type, enable_fine_tuning):
    if model_type == ImageNetModelsPretrained.DENSENET.value or model_type == ImageNetModelsPretrained.MULTILABEL_ALL_600.value:
        logging.info("image model with densenet model")
        vocab_size = 600

        image_model = models.densenet201(pretrained=True)
        image_model.classifier = nn.Linear(image_model.classifier.in_features, vocab_size)

        encoder_dim = image_model.classifier.in_features

        if model_type == ImageNetModelsPretrained.MULTILABEL_ALL_600.value:
            logging.info("image model with densenet model (all) with multi-label classification 600")
            checkpoint = torch.load('experiments/results/classification_densenet_modifiedrsicd_600.pth.tar')
            image_model.load_state_dict(checkpoint['model'])

        return DenseNetFeatureAndAttrExtractor(image_model), encoder_dim
    else:
        raise Exception("not implemented extractor for the other types")


class DenseNetFeatureAndAttrExtractor(nn.Module):
    def __init__(self, image_model):
        super().__init__()
        self.image_model = image_model

    def forward(self, x):
        modules_features = list(self.image_model.children())[:-1]
        features_extractor = nn.Sequential(*modules_features)

        features = features_extractor(x)
        attrs = self.image_model(x)

        return features, attrs


class FutureAndAttrEncoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, model_type, encoded_image_size=14, enable_fine_tuning=False):
        super(FutureAndAttrEncoder, self).__init__()

        self.enc_image_size = encoded_image_size

        self.extractor, self.encoder_dim = get_image_extractor(model_type, enable_fine_tuning)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size))

        # Disable calculation of all gradients

        for p in self.extractor.image_model.parameters():
            p.requires_grad = False

        # Enable calculation of some gradients for fine tuning
        self.fine_tune(enable_fine_tuning)

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        features, attrs = self.extractor(
            images)  # (batch_size, 2048, image_size/32, image_size/32)

        # (batch_size, 2048, encoded_image_size, encoded_image_size)
        features = self.adaptive_pool(features)
        # (batch_size, encoded_image_size, encoded_image_size, 2048)
        # (later on the intermidiate dims are flatten: (prepare_inputs)
        # (batch_size, encoded_image_size*encoded_image_size, 2048)
        features = features.permute(0, 2, 3, 1)

        return features, attrs

    def fine_tune(self, enable_fine_tuning):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.extractor.image_model.children()):  # [5:]:#toda!!!
            for p in c.parameters():
                p.requires_grad = enable_fine_tuning


class ContinuousDecoderWithAttentionAndImage(DecoderWithAttention):
    """
    Decoder.
    """

    def __init__(
            self, attention_dim, embedding_type, embed_dim, decoder_dim, vocab_size, token_to_id, encoder_dim=2048,
            dropout=0.5):

        super(ContinuousDecoderWithAttentionAndImage, self).__init__(attention_dim, embedding_type,
                                                                     embed_dim, decoder_dim, vocab_size, token_to_id, encoder_dim, dropout)

        # linear layer to find representation of image
        self.represent_image = nn.Linear(encoder_dim, embed_dim)
        self.image_embedding = None

        # replace softmax layer with embedding layer
        self.fc = nn.Linear(decoder_dim, embed_dim)

    def init_hidden_state(self, encoder_attr):

        h = encoder_attr  # (batch_size, decoder_dim) 512 units

        self.image_embedding = encoder_attr  # 300 512

        return h, h


class ContinuousAttentionImageAttr600Model(ContinuousAttentionModel):

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

        # if (self.args.embedding_type not in [embedding.value for embedding in EmbeddingsType]):
        #     raise ValueError(
        #         "Continuous model should use pretrained embeddings...")

        self.encoder = FutureAndAttrEncoder(self.args.image_model_type,
                                            enable_fine_tuning=self.args.fine_tune_encoder)

        self.decoder = ContinuousDecoderWithAttentionAndImage(
            encoder_dim=self.encoder.encoder_dim,
            attention_dim=self.args.attention_dim,
            decoder_dim=self.args.decoder_dim,
            embedding_type=self.args.embedding_type,
            embed_dim=self.args.embed_dim,
            vocab_size=self.vocab_size,
            token_to_id=self.token_to_id,
            dropout=self.args.dropout
        )

        self.decoder.normalize_embeddings(self.args.no_normalization)

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

    def _prepare_inputs_to_forward_pass(self, imgs, caps, caption_lengths):
        imgs = imgs.to(self.device)
        caps = caps.to(self.device)
        caption_lengths = caption_lengths.to(self.device)

        # encoder #TODO: MUDAR
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

        return encoder_out, caps_sorted, caption_lengths

    def _predict(self, encoder_out, caps, caption_lengths):

        encoder_features, encoder_attrs = encoder_out

        batch_size = encoder_features.size(0)
        num_pixels = encoder_features.size(1)

        # Create tensors to hold word predicion scores and alphas
        all_predictions = torch.zeros(batch_size,  max(
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
                caps[: batch_size_t, t],
                encoder_features[: batch_size_t],
                h[: batch_size_t],
                c[: batch_size_t])

            all_predictions[:batch_size_t, t, :] = predictions
            all_alphas[:batch_size_t, t, :] = alpha

        return {"predictions": all_predictions, "alphas": all_alphas}

    def generate_output_embedding(self, input_embedding, encoder_out, h, c):
        predictions, h, c, _ = self.decoder(
            input_embedding, encoder_out, h, c)

        current_output_index = self._convert_prediction_to_output(predictions)

        return predictions, current_output_index, h, c

    def inference_with_greedy(self, image, n_solutions=0):
        with torch.no_grad():  # no need to track history

            decoder_sentence = []

            input_word = torch.tensor([self.token_to_id[START_TOKEN]])

            i = 1

            encoder_features, encoder_attrs = self.encoder(image)
            encoder_features = encoder_features.view(encoder_features.size(0), -1, encoder_features.size(-1))  # flatten

            h, c = self.decoder.init_hidden_state(encoder_attrs)

            while True:

                scores, h, c = self.generate_output_index(
                    input_word, encoder_features, h, c)

                sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)

                current_output_index = sorted_indices[0]

                current_output_token = self.id_to_token[current_output_index.item(
                )]

                decoder_sentence.append(current_output_token)

                if current_output_token == END_TOKEN:
                    # ignore end_token
                    decoder_sentence = decoder_sentence[:-1]
                    break

                if i >= self.max_len-1:  # until 35
                    break

                input_word[0] = current_output_index.item()

                i += 1

            generated_sentence = " ".join(decoder_sentence)
            print("\ngenerated sentence:", generated_sentence)

            return generated_sentence  # input_caption

    def inference_with_beamsearch(self, image, n_solutions=3):

        def compute_probability(seed_text, seed_prob, sorted_scores, index, current_text):
            return (seed_prob*len(seed_text) + np.log(sorted_scores[index].item())) / (len(seed_text)+1)

        def compute_perplexity(seed_text, seed_prob, sorted_scores, index, current_text):
            current_text = ' '.join(current_text)
            tokens = self.language_model_tokenizer.encode(current_text)

            input_ids = torch.tensor(tokens).unsqueeze(0)
            with torch.no_grad():
                outputs = self.language_model(input_ids, labels=input_ids)
                loss, logits = outputs[:2]

            return math.exp(loss / len(tokens))

        def compute_sim2image(seed_text, seed_prob, sorted_scores, index, current_text):
            n_tokens = len(current_text)
            tokens_ids = torch.zeros(1, n_tokens)
            for i in range(n_tokens):
                token = current_text[i]
                tokens_ids[0, i] = self.token_to_id[token]

            tokens_embeddings = self.decoder.embedding(tokens_ids.long()).to(self.device)

            sentence_mean = torch.mean(tokens_embeddings, dim=1)
            images_embedding = self.decoder.image_embedding

            return torch.cosine_similarity(sentence_mean, images_embedding)

        def compute_perplexity_with_sim2image():
            return 0

        def generate_n_solutions(seed_text, seed_prob, encoder_features,  h, c,  n_solutions):
            last_token = seed_text[-1]

            if last_token == END_TOKEN:
                return [(seed_text, seed_prob, h, c)]

            top_solutions = []
            scores, h, c = self.generate_output_index(
                torch.tensor([self.token_to_id[last_token]]), encoder_features, h, c)

            sorted_scores, sorted_indices = torch.sort(
                scores, descending=True, dim=-1)

            for index in range(n_solutions):
                text = seed_text + [self.id_to_token[sorted_indices[index].item()]]
                # beam search taking into account lenght of sentence
                # prob = (seed_prob*len(seed_text) + np.log(sorted_scores[index].item()) / (len(seed_text)+1))
                text_score = compute_score(seed_text, seed_prob, sorted_scores, index, text)
                top_solutions.append((text, text_score, h, c))

            return top_solutions

        def get_most_probable(candidates, n_solutions, is_to_reverse):
            return sorted(candidates, key=operator.itemgetter(1), reverse=is_to_reverse)[:n_solutions]

        with torch.no_grad():
            encoder_features, encoder_attrs = self.encoder(image)
            encoder_features = encoder_features.view(encoder_features.size(0), -1, encoder_features.size(-1))  # flatten

            h, c = self.decoder.init_hidden_state(encoder_attrs)

            top_solutions = [([START_TOKEN], 0.0, h, c)]

            if self.args.decodying_type == DecodingType.BEAM.value:
                compute_score = compute_probability
                is_to_reverse = True

            elif self.args.decodying_type == DecodingType.BEAM_PERPLEXITY.value:
                compute_score = compute_perplexity
                is_to_reverse = False

            elif self.args.decodying_type == DecodingType.BEAM_SIM2IMAGE.value:
                compute_score = compute_sim2image

            elif self.args.decodying_type == DecodingType.BEAM_PERPLEXITY_SIM2IMAGE.value:
                compute_score = compute_perplexity_with_sim2image

            else:
                raise Exception("not available any other decoding type")

            for _ in range(self.max_len):
                candidates = []
                for sentence, prob, h, c in top_solutions:
                    candidates.extend(generate_n_solutions(
                        sentence, prob, encoder_features, h, c,  n_solutions))

                top_solutions = get_most_probable(candidates, n_solutions, is_to_reverse)

            # print("top solutions", [(text, prob)
            #                         for text, prob, _, _ in top_solutions])
            best_tokens, prob, h, c = top_solutions[0]

            if best_tokens[0] == START_TOKEN:
                best_tokens = best_tokens[1:]
            if best_tokens[-1] == END_TOKEN:
                best_tokens = best_tokens[:-1]
            best_sentence = " ".join(best_tokens)

            print("\nbeam decoded sentence:", best_sentence)
            return best_sentence
