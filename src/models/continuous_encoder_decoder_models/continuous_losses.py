from enum import Enum
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from torch.nn import functional
from torch import nn
from utils.utils import get_pack_padded_sequences, sink


class ContinuousLossesType(Enum):
    COSINE = "cosine"
    MARGIN = "margin"
    MARGIN_SYN_DISTANCE = "margin_syn_distance"
    MARGIN_SYN_SIMILARITY = "margin_syn_similarity"
    SMOOTHL1 = "smoothl1"
    SMOOTHL1_TRIPLET = "smoothl1_triplet"
    SMOOTHL1_TRIPLET_DIFF = "smoothl1_triplet_diff"
    SMOOTHL1_AVG_SENTENCE = "smoothl1_avg_sentence"
    SMOOTHL1_TRIPLET_AVG_SENTENCE = "smoothl1_triplet_avg_sentence"
    SMOOTHL1_AVG_SENTENCE_BSCORE = "smoothl1_avg_sentence_with_bscore"
    SMOOTHL1_AVG_SENTENCE_AND_INPUT = "smoothl1_avg_sentence_and_input_loss"
    SMOOTHL1_AVG_SENTENCE_AND_INPUTS = "smoothl1_avg_sentence_and_inputs_loss"
    SMOOTHL1_AVG_SENTENCE_AND_INPUTS_NORMALIZED = "smoothl1_avg_sentence_and_inputs_normalized"
    SMOOTHL1_TRIPLET_AVG_SENTENCE_AND_INPUTS = "smoothl1_triplet_avg_sentence_and_inputs"
    SMOOTHL1_SINK_SENTENCE = "smoothl1_sink_sentence"
    COS_AVG_SENTENCE_AND_INPUTS = "cos_avg_sentence_and_inputs_loss"
    COS_AVG_SENTENCE_AND_INPUTS_WEIGHTED = "cos_avg_sentence_and_inputs_loss_weightedbyhalf"
    COS_AVG_SENTENCE = "cos_avg_sentence"
    COS_AVG_SENTENCE_AND_INPUT = "cos_avg_sentence_and_input_loss"


class ContinuousLoss():

    def __init__(self, loss_type, device, decoder):
        self.device = device
        self.decoder = decoder

        if loss_type == ContinuousLossesType.COSINE.value:
            self.loss_method = self.cosine_loss
            self.criterion = nn.CosineEmbeddingLoss().to(self.device)

        elif loss_type == ContinuousLossesType.MARGIN.value:
            self.loss_method = self.margin_loss
            self.criterion = nn.TripletMarginLoss(
                margin=1.0, p=2).to(self.device)

        elif loss_type == ContinuousLossesType.MARGIN_SYN_DISTANCE.value:
            self.loss_method = self.margin_syn_distance_loss
            self.criterion = nn.TripletMarginLoss(
                margin=1.0, p=2).to(self.device)

        elif loss_type == ContinuousLossesType.MARGIN_SYN_SIMILARITY.value:
            self.loss_method = self.margin_syn_similarity_loss
            self.margin = 1.0

        elif loss_type == ContinuousLossesType.SMOOTHL1.value:
            self.loss_method = self.smoothl1_loss
            self.criterion = nn.SmoothL1Loss().to(self.device)

        elif loss_type == ContinuousLossesType.SMOOTHL1_TRIPLET.value:
            self.loss_method = self.smoothl1_triplet_loss
            self.criterion = nn.SmoothL1Loss(reduction='none').to(self.device)
            self.margin = 1.0

        elif loss_type == ContinuousLossesType.SMOOTHL1_TRIPLET_DIFF.value:
            self.loss_method = self.smoothl1_triplet_diff_loss
            self.criterion = nn.SmoothL1Loss(reduction='none').to(self.device)
            self.margin = 1.0

        elif loss_type == ContinuousLossesType.SMOOTHL1_AVG_SENTENCE.value:
            self.loss_method = self.smoothl1_avg_sentence_loss
            self.criterion = nn.SmoothL1Loss().to(self.device)

        elif loss_type == ContinuousLossesType.SMOOTHL1_TRIPLET_AVG_SENTENCE.value:
            self.loss_method = self.smoothl1_triplet_avg_sentence_loss
            self.criterion = nn.SmoothL1Loss(reduction='none').to(self.device)
            self.margin = 1.0

        elif loss_type == ContinuousLossesType.SMOOTHL1_AVG_SENTENCE_BSCORE.value:
            self.loss_method = self.smoothl1_avg_sentence_with_bscore_loss
            self.criterion = nn.SmoothL1Loss().to(self.device)

        elif loss_type == ContinuousLossesType.SMOOTHL1_AVG_SENTENCE_AND_INPUT.value:
            self.loss_method = self.smoothl1_avg_sentence_and_input_loss
            self.criterion = nn.SmoothL1Loss().to(self.device)

        elif loss_type == ContinuousLossesType.SMOOTHL1_AVG_SENTENCE_AND_INPUTS.value:
            self.loss_method = self.smoothl1_avg_sentence_and_inputs_loss
            self.criterion = nn.SmoothL1Loss().to(self.device)

        elif loss_type == ContinuousLossesType.SMOOTHL1_AVG_SENTENCE_AND_INPUTS_NORMALIZED.value:
            self.loss_method = self.smoothl1_avg_sentence_and_inputs_normalized_loss
            self.criterion = nn.SmoothL1Loss().to(self.device)

        elif loss_type == ContinuousLossesType.SMOOTHL1_TRIPLET_AVG_SENTENCE_AND_INPUTS.value:
            self.loss_method = self.smoothl1_triplet_avg_sentence_and_inputs_loss
            self.criterion = nn.SmoothL1Loss().to(self.device)
            self.margin = 1.0

        elif loss_type == ContinuousLossesType.SMOOTHL1_SINK_SENTENCE.value:
            self.loss_method = self.smoothl1_sink_sentence_loss
            self.criterion = nn.SmoothL1Loss().to(self.device)

        elif loss_type == ContinuousLossesType.COS_AVG_SENTENCE_AND_INPUTS.value:
            self.loss_method = self.cos_avg_sentence_and_inputs_loss
            self.criterion = nn.CosineEmbeddingLoss().to(self.device)

        elif loss_type == ContinuousLossesType.COS_AVG_SENTENCE.value:
            self.loss_method = self.cos_avg_sentence_loss
            self.criterion = nn.CosineEmbeddingLoss().to(self.device)

        elif loss_type == ContinuousLossesType.COS_AVG_SENTENCE_AND_INPUT.value:
            self.loss_method = self.cos_avg_sentence_and_input_loss
            self.criterion = nn.CosineEmbeddingLoss().to(self.device)

    def compute_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths
    ):
        return self.loss_method(predictions, target_embeddings, caption_lengths)

    def cosine_loss(
            self,
            predictions,
            target_embeddings,
            caption_lengths
    ):
        predictions, target_embeddings = get_pack_padded_sequences(predictions, target_embeddings, caption_lengths)
        y = torch.ones(target_embeddings.shape[0]).to(self.device)

        return self.criterion(predictions, target_embeddings,  y)

    def margin_syn_distance_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths
    ):
        predictions, target_embeddings = get_pack_padded_sequences(predictions, target_embeddings, caption_lengths)
        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)

        orthogonal_component = (predictions - torch.sum(predictions*target_embeddings,
                                                        dim=1).unsqueeze(1) * target_embeddings)

        orthogonal_negative_examples = torch.nn.functional.normalize(orthogonal_component, p=2, dim=-1)

        return self.criterion(predictions, target_embeddings, orthogonal_negative_examples.to(self.device))

    def margin_syn_similarity_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths
    ):
        predictions, target_embeddings = get_pack_padded_sequences(predictions, target_embeddings, caption_lengths)
        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)

        orthogonal_component = (predictions - torch.sum(predictions*target_embeddings,
                                                        dim=1).unsqueeze(1) * target_embeddings)

        orthogonal_negative_examples = torch.nn.functional.normalize(orthogonal_component, p=2, dim=-1)

        sim_to_negative = torch.sum(predictions*orthogonal_negative_examples, dim=1)
        sim_to_target = torch.sum(predictions*target_embeddings, dim=1)

        loss = torch.clamp(self.margin + sim_to_negative - sim_to_target, min=0).mean()
        return loss

    def margin_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths
    ):
        predictions, target_embeddings = get_pack_padded_sequences(predictions, target_embeddings, caption_lengths)
        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)

        negative_examples = torch.zeros(target_embeddings.size()[0], target_embeddings.size()[1])

        pretrained_embedding_matrix = self.decoder.embedding.weight.data
        for i in range(len(target_embeddings)):

            diff = predictions[i] - target_embeddings[i]

            target_similarity_to_embeddings = functional.cosine_similarity(diff.unsqueeze_(0),
                                                                           pretrained_embedding_matrix)

            top_scores, top_indices = torch.topk(target_similarity_to_embeddings, k=1, dim=0)
            id_most_informative_negative = top_indices[0]
            informative_negative_embedding = pretrained_embedding_matrix[id_most_informative_negative]

            negative_examples[i, :] = informative_negative_embedding

        return self.criterion(predictions, target_embeddings,  negative_examples.to(self.device))

    def smoothl1_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths
    ):
        predictions, target_embeddings = get_pack_padded_sequences(predictions, target_embeddings, caption_lengths)
        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)

        return self.criterion(predictions, target_embeddings)

    def smoothl1_triplet_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths
    ):
        predictions, target_embeddings = get_pack_padded_sequences(predictions, target_embeddings, caption_lengths)
        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)

        orthogonal_component = (predictions - torch.sum(predictions*target_embeddings,
                                                        dim=1).unsqueeze(1) * target_embeddings)

        orthogonal_negative_examples = torch.nn.functional.normalize(orthogonal_component, p=2, dim=-1)

        # apply distance of smoothl1
        dist_to_negative = self.criterion(predictions, orthogonal_negative_examples)
        dist_to_target = self.criterion(predictions, target_embeddings)

        loss = torch.clamp(self.margin + dist_to_target - dist_to_negative, min=0).mean()
        return loss

    def smoothl1_triplet_diff_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths
    ):
        predictions, target_embeddings = get_pack_padded_sequences(predictions, target_embeddings, caption_lengths)
        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)

        diff_component = (predictions - target_embeddings)

        diff_negative_examples = torch.nn.functional.normalize(diff_component, p=2, dim=-1)

        # apply distance of smoothl1
        dist_to_negative = self.criterion(predictions, diff_negative_examples)
        dist_to_target = self.criterion(predictions, target_embeddings)

        loss = torch.clamp(self.margin + dist_to_target - dist_to_negative, min=0).mean()
        return loss

    def sentence_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths
    ):
        word_losses = 0.0

        n_sentences = predictions.size()[0]
        for i in range(n_sentences):  # iterate by sentence
            preds_without_padd = predictions[i, :caption_lengths[i], :]
            targets_without_padd = target_embeddings[i, :caption_lengths[i], :]

            y = torch.ones(targets_without_padd.shape[0])

            # word-level loss
            word_losses += self.criterion(
                preds_without_padd,
                targets_without_padd,
                y
            )

        word_loss = word_losses/n_sentences

        return word_loss

    def smoothl1_avg_sentence_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths
    ):
        word_losses = 0.0
        sentence_losses = 0.0
        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)

        n_sentences = predictions.size()[0]
        for i in range(n_sentences):  # iterate by sentence
            preds_without_padd = predictions[i, :caption_lengths[i], :]
            targets_without_padd = target_embeddings[i, :caption_lengths[i], :]

            # word-level loss
            word_losses += self.criterion(
                preds_without_padd,
                targets_without_padd
            )

            # sentence-level loss
            sentence_mean_pred = torch.mean(preds_without_padd, dim=0)  # ver a dim
            sentece_mean_target = torch.mean(targets_without_padd, dim=0)

            sentence_losses += self.criterion(
                sentence_mean_pred,
                sentece_mean_target
            )

        word_loss = word_losses/n_sentences
        sentence_loss = sentence_losses/n_sentences

        loss = word_loss + sentence_loss

        return loss

    def smoothl1_avg_sentence_with_bscore_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths
    ):
        word_losses = 0.0
        sentence_losses = 0.0
        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)

        n_sentences = predictions.size()[0]

        def sim_matrix(a, b, eps=1e-8):
            """
            added eps for numerical stability
            """
            a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
            a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
            b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
            sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))

            return sim_mt

        for i in range(n_sentences):  # iterate by sentence
            preds_without_padd = predictions[i, :caption_lengths[i], :]
            targets_without_padd = target_embeddings[i, :caption_lengths[i], :]

            # word-level loss
            word_losses += self.criterion(
                preds_without_padd,
                targets_without_padd
            )

            pairwise_cosine_similarity = sim_matrix(preds_without_padd, targets_without_padd)
            # cos = cosine_similarity(preds_without_padd, targets_without_padd)

            maximum_similarity, _ = torch.max(pairwise_cosine_similarity, dim=1)
            # torch.sum*pesos/dividindo pelos pesos

            sentence_losses += (1-torch.mean(maximum_similarity))

        word_loss = word_losses/n_sentences
        sentence_loss = sentence_losses/n_sentences

        loss = word_loss + sentence_loss

        return loss

    def smoothl1_triplet_avg_sentence_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths
    ):
        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)
        word_losses = 0.0
        sentence_losses = 0.0

        n_sentences = predictions.size()[0]
        for i in range(n_sentences):  # iterate by sentence
            preds_without_padd = predictions[i, :caption_lengths[i], :]
            targets_without_padd = target_embeddings[i, :caption_lengths[i], :]

            orthogonal_component = (preds_without_padd - torch.sum(preds_without_padd*targets_without_padd,
                                                                   dim=1).unsqueeze(1) * targets_without_padd)

            orthogonal_negative_examples = torch.nn.functional.normalize(orthogonal_component, p=2, dim=-1)

            dist_to_negative = self.criterion(preds_without_padd, orthogonal_negative_examples)
            dist_to_target = self.criterion(preds_without_padd, targets_without_padd)

            word_losses += torch.clamp(self.margin + dist_to_target - dist_to_negative, min=0).mean()

            # sentence-level loss
            sentence_mean_pred = torch.mean(preds_without_padd, dim=0)  # ver a dim
            sentece_mean_target = torch.mean(targets_without_padd, dim=0)
            sentence_mean_ortogonal = torch.mean(orthogonal_negative_examples, dim=0)

            dist_to_negative = self.criterion(sentence_mean_pred, sentence_mean_ortogonal)
            dist_to_target = self.criterion(sentence_mean_pred, sentece_mean_target)

            sentence_losses += torch.clamp(self.margin + dist_to_target - dist_to_negative, min=0).mean()

        word_loss = word_losses/n_sentences
        sentence_loss = sentence_losses/n_sentences

        loss = word_loss + sentence_loss

        return loss

    def smoothl1_avg_sentence_and_input_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths
    ):
        word_losses = 0.0  # pred_against_target_loss; #pred_sentence_again_target_sentence;"pred_sentence_agains_image
        sentence_losses = 0.0
        input_losses = 0.0

        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)
        images_embedding = self.decoder.image_embedding

        n_sentences = predictions.size()[0]
        for i in range(n_sentences):  # iterate by sentence
            preds_without_padd = predictions[i, :caption_lengths[i], :]
            targets_without_padd = target_embeddings[i, :caption_lengths[i], :]

            # word-level loss   (each prediction against each target)
            word_losses += self.criterion(
                preds_without_padd,
                targets_without_padd
            )

            # sentence-level loss (sentence predicted agains target sentence)
            sentence_mean_pred = torch.mean(preds_without_padd, dim=0)  # ver a dim
            sentece_mean_target = torch.mean(targets_without_padd, dim=0)

            sentence_losses += self.criterion(
                sentence_mean_pred,
                sentece_mean_target
            )

            image_embedding = torch.nn.functional.normalize(images_embedding[i], p=2, dim=-1)

            # input loss (sentence predicted against input image)
            input_losses += self.criterion(
                sentence_mean_pred,
                image_embedding
            )

        word_loss = word_losses/n_sentences
        sentence_loss = sentence_losses/n_sentences
        input_loss = input_losses/n_sentences

        loss = word_loss + sentence_loss + input_loss

        return loss

    def smoothl1_avg_sentence_and_inputs_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths
    ):
        word_losses = 0.0  # pred_against_target_loss; #pred_sentence_again_target_sentence;"pred_sentence_agains_image
        sentence_losses = 0.0
        input1_losses = 0.0
        input2_losses = 0.0

        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)
        images_embedding = self.decoder.image_embedding

        n_sentences = predictions.size()[0]
        for i in range(n_sentences):  # iterate by sentence
            preds_without_padd = predictions[i, :caption_lengths[i], :]
            targets_without_padd = target_embeddings[i, :caption_lengths[i], :]

            # word-level loss   (each prediction against each target)
            word_losses += self.criterion(
                preds_without_padd,
                targets_without_padd
            )

            # sentence-level loss (sentence predicted agains target sentence)
            sentence_mean_pred = torch.mean(preds_without_padd, dim=0)  # ver a dim
            sentece_mean_target = torch.mean(targets_without_padd, dim=0)

            sentence_losses += self.criterion(
                sentence_mean_pred,
                sentece_mean_target
            )

            image_embedding = images_embedding[i]

            # 1º input loss (sentence predicted against input image)
            input1_losses += self.criterion(
                sentence_mean_pred,
                image_embedding
            )

            # 2º input loss (image predicted against targe)
            input2_losses += self.criterion(
                image_embedding,
                sentece_mean_target
            )

        word_loss = word_losses/n_sentences
        sentence_loss = sentence_losses/n_sentences
        input1_loss = input1_losses/n_sentences
        input2_loss = input2_losses/n_sentences

        loss = word_loss + sentence_loss + input1_loss + input2_loss

        return loss

    def smoothl1_avg_sentence_and_inputs_normalized_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths
    ):
        word_losses = 0.0  # pred_against_target_loss; #pred_sentence_again_target_sentence;"pred_sentence_agains_image
        sentence_losses = 0.0
        input1_losses = 0.0
        input2_losses = 0.0

        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)
        images_embedding = self.decoder.image_embedding

        n_sentences = predictions.size()[0]
        for i in range(n_sentences):  # iterate by sentence
            preds_without_padd = predictions[i, :caption_lengths[i], :]
            targets_without_padd = target_embeddings[i, :caption_lengths[i], :]

            # word-level loss   (each prediction against each target)
            word_losses += self.criterion(
                preds_without_padd,
                targets_without_padd
            )

            # sentence-level loss (sentence predicted agains target sentence)
            sentence_mean_pred = torch.mean(preds_without_padd, dim=0)  # ver a dim
            sentece_mean_target = torch.mean(targets_without_padd, dim=0)

            sentence_losses += self.criterion(
                sentence_mean_pred,
                sentece_mean_target
            )

            image_embedding = torch.nn.functional.normalize(images_embedding[i], p=2, dim=-1)

            # 1º input loss (sentence predicted against input image)
            input1_losses += self.criterion(
                sentence_mean_pred,
                image_embedding
            )

            # 2º input loss (sentence predicted against input image)
            input2_losses += self.criterion(
                image_embedding,
                sentece_mean_target
            )

        word_loss = word_losses/n_sentences
        sentence_loss = sentence_losses/n_sentences
        input1_loss = input1_losses/n_sentences
        input2_loss = input2_losses/n_sentences

        loss = word_loss + sentence_loss + input1_loss + input2_loss

        return loss

    def smoothl1_triplet_avg_sentence_and_inputs_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths
    ):
        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)
        word_losses = 0.0
        sentence_losses = 0.0
        input1_losses = 0.0
        input2_losses = 0.0

        n_sentences = predictions.size()[0]
        images_embedding = self.decoder.image_embedding
        parameter = 10

        for i in range(n_sentences):  # iterate by sentence
            preds_without_padd = predictions[i, :caption_lengths[i], :]
            targets_without_padd = target_embeddings[i, :caption_lengths[i], :]

            orthogonal_component = (preds_without_padd - torch.sum(preds_without_padd*targets_without_padd,
                                                                   dim=1).unsqueeze(1) * targets_without_padd)
            orthogonal_negative_examples = torch.nn.functional.normalize(orthogonal_component, p=2, dim=-1)

            dist_to_negative = self.criterion(preds_without_padd, orthogonal_negative_examples)
            dist_to_target = self.criterion(preds_without_padd, targets_without_padd)

            word_losses += torch.log(1+torch.exp(parameter*(dist_to_target-dist_to_negative)))

            # sentence-level loss
            sentence_mean_pred = torch.nn.functional.normalize(
                torch.mean(preds_without_padd, dim=0), p=2, dim=-1)  # ver a dim
            sentece_mean_target = torch.nn.functional.normalize(torch.mean(targets_without_padd, dim=0), p=2, dim=-1)

            sentence_mean_ortogonal = (sentence_mean_pred - torch.sum(sentence_mean_pred.unsqueeze(0)
                                                                      * sentece_mean_target.unsqueeze(0), dim=1) * sentece_mean_target)
            orthogonal_negative_examples = torch.nn.functional.normalize(sentence_mean_ortogonal, p=2, dim=-1)

            dist_to_negative = self.criterion(sentence_mean_pred, sentence_mean_ortogonal)
            dist_to_target = self.criterion(sentence_mean_pred, sentece_mean_target)

            sentence_losses += torch.log(1+torch.exp(parameter*(dist_to_target-dist_to_negative)))

            # 1º input loss (sentence predicted against input image)
            # input1_losses += self.criterion(
            #     sentence_mean_pred,
            #     images_embedding[i]
            # )

            image_embedding = torch.nn.functional.normalize(images_embedding[i], p=2, dim=-1)

            orthogonal_component = (sentence_mean_pred - torch.sum(sentence_mean_pred.unsqueeze(0)
                                                                   * image_embedding.unsqueeze(0), dim=1) * image_embedding)
            image_negative_example = torch.nn.functional.normalize(orthogonal_component, p=2, dim=-1)

            dist_to_negative = self.criterion(sentence_mean_pred, image_negative_example)
            dist_to_target = self.criterion(sentence_mean_pred, image_embedding)

            input1_losses += torch.log(1+torch.exp(parameter*(dist_to_target-dist_to_negative)))

            # 2º input loss (sentence predicted against input image)
            # input2_losses += self.criterion(
            #     images_embedding[i],
            #     sentece_mean_target
            # )
            dist_to_negative = self.criterion(image_embedding, sentence_mean_ortogonal)
            dist_to_target = self.criterion(image_embedding, sentece_mean_target)

            input2_losses += torch.log(1+torch.exp(parameter*(dist_to_target-dist_to_negative)))

        word_loss = word_losses/n_sentences
        sentence_loss = sentence_losses/n_sentences
        input1_loss = input1_losses/n_sentences
        input2_loss = input2_losses/n_sentences

        loss = word_loss + sentence_loss + input1_loss + input2_loss

        return loss

    def smoothl1_sink_sentence_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths
    ):

        def dmat(x, y):
            mmp1 = torch.stack([x] * x.size()[0])
            mmp2 = torch.stack([y] * y.size()[0]).transpose(0, 1)

            return torch.sum((mmp1 - mmp2) ** 2, 2).squeeze()

        word_losses = 0.0
        sentence_losses = 0.0
        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)

        n_sentences = predictions.size()[0]
        for i in range(n_sentences):  # iterate by sentence
            preds_without_padd = predictions[i, :caption_lengths[i], :]
            targets_without_padd = target_embeddings[i, :caption_lengths[i], :]

            # word-level loss
            word_losses += self.criterion(
                preds_without_padd,
                targets_without_padd
            )

            # sentence-level loss
            sentence_losses += sink(dmat(preds_without_padd, targets_without_padd), reg=10, cuda=True)

        word_loss = word_losses/n_sentences
        sentence_loss = sentence_losses/n_sentences

        loss = word_loss + sentence_loss

        return loss

    def cos_avg_sentence_and_inputs_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths
    ):
        word_losses = 0.0  # pred_against_target_loss; #pred_sentence_again_target_sentence;"pred_sentence_agains_image
        sentence_losses = 0.0
        input1_losses = 0.0
        input2_losses = 0.0

        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)
        images_embedding = self.decoder.image_embedding

        n_sentences = predictions.size()[0]
        for i in range(n_sentences):  # iterate by sentence
            preds_without_padd = predictions[i, :caption_lengths[i], :]
            targets_without_padd = target_embeddings[i, :caption_lengths[i], :]
            y = torch.ones(targets_without_padd.shape[0]).to(self.device)

            # word-level loss   (each prediction against each target)
            word_losses += self.criterion(
                preds_without_padd,
                targets_without_padd,
                y
            )

            # sentence-level loss (sentence predicted agains target sentence)
            sentence_mean_pred = torch.mean(preds_without_padd, dim=0).unsqueeze(0)  # ver a dim
            sentece_mean_target = torch.mean(targets_without_padd, dim=0).unsqueeze(0)

            y = torch.ones(1).to(self.device)

            sentence_losses += self.criterion(
                sentence_mean_pred,
                sentece_mean_target,
                y
            )

            image_embedding = images_embedding[i].unsqueeze(0)

            # 1º input loss (sentence predicted against input image)
            input1_losses += self.criterion(
                sentence_mean_pred,
                image_embedding,
                y
            )

            # 2º input loss (sentence predicted against input image)
            input2_losses += self.criterion(
                image_embedding,
                sentece_mean_target,
                y
            )

        word_loss = word_losses/n_sentences
        sentence_loss = sentence_losses/n_sentences
        input1_loss = input1_losses/n_sentences
        input2_loss = input2_losses/n_sentences

        loss = word_loss + sentence_loss + input1_loss + input2_loss

        return loss

    def cos_avg_sentence_and_input_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths
    ):
        word_losses = 0.0  # pred_against_target_loss; #pred_sentence_again_target_sentence;"pred_sentence_agains_image
        sentence_losses = 0.0
        input1_losses = 0.0

        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)
        images_embedding = self.decoder.image_embedding

        n_sentences = predictions.size()[0]
        for i in range(n_sentences):  # iterate by sentence
            preds_without_padd = predictions[i, :caption_lengths[i], :]
            targets_without_padd = target_embeddings[i, :caption_lengths[i], :]
            y = torch.ones(targets_without_padd.shape[0]).to(self.device)

            # word-level loss   (each prediction against each target)
            word_losses += self.criterion(
                preds_without_padd,
                targets_without_padd,
                y
            )

            # sentence-level loss (sentence predicted agains target sentence)
            sentence_mean_pred = torch.mean(preds_without_padd, dim=0).unsqueeze(0)  # ver a dim
            sentece_mean_target = torch.mean(targets_without_padd, dim=0).unsqueeze(0)

            y = torch.ones(1).to(self.device)

            sentence_losses += self.criterion(
                sentence_mean_pred,
                sentece_mean_target,
                y
            )

            image_embedding = images_embedding[i].unsqueeze(0)

            # 1º input loss (sentence predicted against input image)
            input1_losses += self.criterion(
                sentence_mean_pred,
                image_embedding,
                y
            )

        word_loss = word_losses/n_sentences
        sentence_loss = sentence_losses/n_sentences
        input1_loss = input1_losses/n_sentences

        loss = word_loss + sentence_loss + input1_loss

        return loss

    def cos_avg_sentence_loss(
        self,
        predictions,
        target_embeddings,
        caption_lengths
    ):
        word_losses = 0.0  # pred_against_target_loss; #pred_sentence_again_target_sentence;"pred_sentence_agains_image
        sentence_losses = 0.0

        predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)

        n_sentences = predictions.size()[0]
        for i in range(n_sentences):  # iterate by sentence
            preds_without_padd = predictions[i, :caption_lengths[i], :]
            targets_without_padd = target_embeddings[i, :caption_lengths[i], :]
            y = torch.ones(targets_without_padd.shape[0]).to(self.device)

            # word-level loss   (each prediction against each target)
            word_losses += self.criterion(
                preds_without_padd,
                targets_without_padd,
                y
            )

            # sentence-level loss (sentence predicted agains target sentence)
            sentence_mean_pred = torch.mean(preds_without_padd, dim=0).unsqueeze(0)  # ver a dim
            sentece_mean_target = torch.mean(targets_without_padd, dim=0).unsqueeze(0)

            y = torch.ones(1).to(self.device)

            sentence_losses += self.criterion(
                sentence_mean_pred,
                sentece_mean_target,
                y
            )

        word_loss = word_losses/n_sentences
        sentence_loss = sentence_losses/n_sentences

        loss = word_loss + sentence_loss

        return loss
