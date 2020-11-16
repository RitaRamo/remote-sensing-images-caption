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
from utils.optimizer import get_optimizer, clip_gradient


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


class ContinuousEncoderDecoderProbSimModel(ContinuousEncoderDecoderModel):

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

        if self.args.grad_norm:
            self.loss_weight_ce = torch.ones(
                1, requires_grad=True, device=self.device, dtype=torch.float
            )
            self.loss_weight_sim = torch.ones(
                1, requires_grad=True, device=self.device, dtype=torch.float
            )
            self.gradnorm_optimizer = torch.optim.Adam(
                [self.loss_weight_ce, self.loss_weight_sim],
                lr=0.025,
            )
            self.gradnorm_loss = nn.L1Loss().to(self.device)
        else:
            self.loss_weight_ce = torch.ones(
                1, device=self.device, dtype=torch.float
            )
            self.loss_weight_sim = torch.ones(
                1, device=self.device, dtype=torch.float
            )
        self.initial = False
        print("AQUI loss weights ce", self.loss_weight_ce.item())
        print("AWUI loss weights sim", self.loss_weight_sim.item())

    def _define_loss_criteria(self):
        self.criterion_ce = nn.CrossEntropyLoss().to(self.device)
        self.criterion_sim = nn.CosineEmbeddingLoss().to(self.device)

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

        y = torch.ones(target_embeddings.data.shape[0]).to(self.device)

        loss_sim = self.criterion_sim(predictions_embeddings.data, target_embeddings.data, y)

        if self.initial == False:
            self.initial = True
            self.initial_ce_loss = loss_ce
            self.initial_sim_loss = loss_sim

        return loss_ce, loss_sim

    def val_step(self, imgs, caps_input, cap_len, all_captions):
        (loss_ce, loss_sim), hypotheses, references_without_padding = super().val_step(imgs, caps_input, cap_len, all_captions)
        loss = self.loss_weight_ce[0].data * loss_ce + self.loss_weight_sim[0].data * loss_sim
        print("weight ce", self.loss_weight_ce[0].data)
        print("weight sim", self.loss_weight_sim[0].data)

        return loss, hypotheses, references_without_padding

    def train_step(self, imgs, caps_input, cap_len):
        encoder_out, caps_sorted, caption_lengths, sort_ind = self._prepare_inputs_to_forward_pass(
            imgs, caps_input, cap_len)

        predict_output = self._predict(
            encoder_out, caps_sorted, caption_lengths)

        loss_ce, loss_sim = self._calculate_loss(
            predict_output, caps_sorted, caption_lengths)

        loss = self.loss_weight_ce[0] * loss_ce + self.loss_weight_sim[0] * loss_sim

        self.decoder_optimizer.zero_grad()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()

        loss.backward(retain_graph=self.args.grad_norm)

        if self.args.grad_norm:
            self.apply_grad_norm(loss_ce, loss_sim)

        # # Clip gradients
        clip_gradient(self.decoder_optimizer, 5.)
        if self.encoder_optimizer is not None:
            clip_gradient(self.encoder_optimizer, 5.)

        # Update weights
        self.decoder_optimizer.step()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.step()

        return loss

    def apply_grad_norm(self, loss_ce, loss_sim):
        # shared_params = self.decoder.named_parameters()
        # print("shared_parms", list(shared_params))
        # shared_params = [
        #     param for param in self.decoder.parameters() if param.requires_grad
        # ]

        # SHARED_PARAMS = [
        #     "decode_step.weight_ih",
        #     "decode_step.weight_hh",
        #     "decode_step.bias_ih",
        #     "decode_step.bias_hh",
        #     "fc.weight",
        #     "fc.bias"
        # ]

        # named_params = dict(self.decoder.named_parameters())

        # shared_params = [
        #     param
        #     for param_name, param in named_params.items()
        #     if param_name in SHARED_PARAMS and param.requires_grad
        # ]

        G1R = torch.autograd.grad(
            loss_ce, self.decoder.fc.parameters(), retain_graph=True, create_graph=True
        )

        G1R_flattened = torch.cat([g.view(-1) for g in G1R])
        G1 = torch.norm(self.loss_weight_ce * G1R_flattened.detach(), 2).unsqueeze(0)

        G2R = torch.autograd.grad(loss_sim, self.decoder.fc.parameters())
        G2R_flattened = torch.cat([g.view(-1) for g in G2R])
        G2 = torch.norm(self.loss_weight_sim * G2R_flattened.detach(), 2).unsqueeze(0)

        # Calculate the average gradient norm across all tasks
        G_avg = torch.div(torch.add(G1, G2), 2)

        # Calculate relative losses
        lhat1 = torch.div(loss_ce.detach(), self.initial_ce_loss)
        lhat2 = torch.div(loss_sim.detach(), self.initial_sim_loss)
        lhat_avg = torch.div(torch.add(lhat1, lhat2), 2)

        # Calculate relative inverse training rates
        inv_rate1 = torch.div(lhat1, lhat_avg)
        inv_rate2 = torch.div(lhat2, lhat_avg)

        # Calculate the gradient norm target for this batch
        C1 = G_avg * (inv_rate1 ** self.args.grad_norm_alpha)
        C2 = G_avg * (inv_rate2 ** self.args.grad_norm_alpha)

        C1 = C1.detach()
        C2 = C2.detach()

        # Backprop and perform an optimization step
        self.gradnorm_optimizer.zero_grad()
        # Calculate the gradnorm loss
        Lgrad = torch.add(self.gradnorm_loss(G1, C1), self.gradnorm_loss(G2, C2))
        Lgrad.backward()
        self.gradnorm_optimizer.step()

        coef = 2 / torch.add(self.loss_weight_ce, self.loss_weight_sim)
        self.loss_weight_ce.data = coef.data * self.loss_weight_ce.data
        self.loss_weight_sim.data = coef.data * self.loss_weight_sim.data

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
