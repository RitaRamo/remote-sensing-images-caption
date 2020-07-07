from enum import Enum


class EmbeddingsType(Enum):
    GLOVE = "glove"
    FASTTEXT = "fasttext"
    CONCATENATE_GLOVE_FASTTEXT = "concatenate_glove_fasttext"
    BERT = "bert"


class ImageNetModelsPretrained(Enum):
    RESNET = "resnet"
    DENSENET = "densenet"
    VGG16 = "vgg16"
    MULTILABEL_ALL = "multilabel_all"  # classification on remote sensing image with all layers unfreezed
    MULTILABEL_ALL_600 = "multilabel_all_600"  # classification on remote sensing image with all layers unfreezed
    MULTILABEL_LAST = "multilabel_last"  # classification on remote sensing image with only last layer unfreezed
    MULTILABEL_ALL_EFFICIENCENET = "efficient_net"


class OptimizerType(Enum):
    ADAM = "adam"
    ADAGRAD = "adagrad"
    ADADELTA = "adadelta"
    SGD = "sgd"


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
    COS_124 = "cos_124"
    COS_14 = "cos_14"
    COS_13 = "cos_13"
    COS_134 = "cos_134"
    COS_SUM_SENTENCE = "cos_sum_sentence"
    COS_AVG_SENTENCE_AND_INPUTS_NORM = "cos_avg_sentence_and_inputs_norm_loss"
    COSINE_NORM = "cosine_norm"
    COS_AVG_SENTENCE_NORM = "cos_avg_sentence_norm"
    COS_13_NORM = "cos_13_norm"
    COSNORM_13_NONORM = "cos_norm_13_nonorm"
    COSNONORM_13_NORM = "cos_nonorm_13_norm"
    COS_134_NORM = "cos_134_norm"
    COS_NONORM_AVG_SENTENCE_NORM = "cos_nonorm_avg_sentence_norm"
    COS_NORM_AVG_SENTENCE_NONORM = "cos_norm_avg_sentence_nonorm"
    COS_NONORM_AVG_SENTENCE_NORM_DIFFICULTY = "cos_nonorm_avg_sentence_norm_difficulty"
    COS_NONORM_DIFFICULTY_AVG_SENTENCE_NORM = "cos_nonorm_difficulty_avg_sentence_norm"


class DecodingType(Enum):
    GREEDY = "greedy"
    GREEDY_EMBEDDING = "greedy_embedding"
    GREEDY_SMOOTHL1 = "greedy_smoothl1"
    BEAM = "beam"
    BEAM_PERPLEXITY = "perplexity"
    BEAM_SIM2IMAGE = "sim2image"
    BEAM_PERPLEXITY_SIM2IMAGE = "perplexity_image"
    POSTPROCESSING_PERPLEXITY = "postprocessing_perplexity"
    BIGRAM_PROB = "bigram_prob"
    POSTPROCESSING_BIGRAM_PROB = "postprocessing_bigramprob"
    BIGRAM_PROB_IMAGE = "bigramprob_and_image"
    BIGRAM_PROB_COS = "bigramprob_and_cos"
