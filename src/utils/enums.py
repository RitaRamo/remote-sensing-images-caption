from enum import Enum


class EarlyStopMode(Enum):
    LOSS = "loss"
    METRIC = "metric"


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
    EFFICIENCENET_RSICD_NOUNSADJS_EMBEDDINGS = "efficient_net_rsicd_na_emb"
    EFFICIENCENET_RSICD_CAPTION_EMBEDDINGS = "efficient_net_rsicd_caption_emb"
    EFFICIENCENET_RSICD_CAPTION_GLOVE_EMBEDDINGS = "efficient_net_rsicd_caption_emb_glove"
    EFFICIENCENET_RSICD_CAPTION_GLOVE_EMBEDDINGS_SMOOTHL1 = "efficient_net_rsicd_caption_emb_glove_smoothl1"
    EFFICIENCENET_RSICD_CAPTION_FASTTEXT_EMBEDDINGS_SMOOTHL1 = "efficient_net_rsicd_caption_emb_fasttext_smoothl1"
    EFFICIENCENET_EMBEDDINGS = "efficient_net_embeddings"
    EFFICIENCENET_UCM = "efficient_net_ucm"
    EFFICIENCENET_FLICKR8K = "efficient_net_flickr8k"
    EFFICIENCENET_IMAGENET = "efficient_net_imagenet"


class OptimizerType(Enum):
    ADAM = "adam"
    ADAGRAD = "adagrad"
    ADADELTA = "adadelta"
    SGD = "sgd"


class Datasets(Enum):
    RSICD = "rsicd"
    RSICD_NEW_SPLITS_TRAIN_VAL = "rsicd_new_splits"
    SIDNEY = "sidney"
    UCM = "ucm"
    FLICKR8K = "flickr8k"


class EvalDatasetType(Enum):
    TRAIN_AND_VAL = "train_and_val"
    VAL = "val"
    TEST = "test"


class ContinuousLossesType(Enum):
    COSINE = "cosine"
    MARGIN = "margin"
    MAX_MARGIN_WORD = "max_margin_word"
    MAX_MARGIN_COS_WORD = "max_margin_cos_loss"
    MARGIN_SYN_DISTANCE = "margin_syn_distance"
    MARGIN_SYN_SIMILARITY = "margin_syn_similarity"
    SMOOTHL1 = "smoothl1"
    SMOOTHL1_SUM_MEAN = "smoothl1_sum_mean"
    TSS_LOSS = "tss_loss"
    L1_LOSS = "l1"
    L1_SUM_MEAN = "l1_sum_mean"
    L2_LOSS = "l2"
    L2_SUM_MEAN = "l2_sum_mean"
    COS_L1 = "cos_l1"
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
    COS_NONORM_AVG_SENTENCE_AND_INPUTS_NORM = "cos_nonorm_avg_sentence_and_inputs_norm_loss"
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
    COS_NONORM_AVG_SENTENCE_NORM_AND_INPUT_NONORM = "cos_nonorm_avg_sentence_norm_and_input_nonorm_loss"
    COS_AVG_SENTENCE75 = "cos_avg_sentence75"
    COS_AVG_SENTENCE50 = "cos_avg_sentence50"
    COS75_AVG_SENTENCE = "cos75_avg_sentence"
    COS_AVG_SENTENCE_AND_INPUTS_W = "cos_avg_sentence_and_inputs_w_loss"
    COS_HDSentence = "cos_hausdorffsentence"
    COS_F1HDSentence = "cos_f1hausdorffsentence"
    COS_D1HDSentence = "cos_d1hausdorffsentence"
    COS_BScoreSentence = "cos_bscoresentence"
    COS_HDSENTENCE_AND_INPUTS = "cos_hausdorffsentence_and_inputs"
    COS_HDSENTENCE_AND_HDINPUTS = "cos_hausdorffsentence_and_hdinputs"
    COS_AVG_HDSENTENCE_AND_AVG_HDINPUTS = "cos_avg_hausdorffsentence_and_avg_hdinputs"
    COS_AVG_HDSENTENCE = "cos_avg_hausdorffsentence"


class DecodingType(Enum):
    GREEDY = "greedy"
    GREEDY_EMBEDDING = "greedy_embedding"
    GREEDY_SMOOTHL1 = "greedy_smoothl1"
    GREEDY_SIM_RANK = "greedy_sim_rank"
    BEAM = "beam"
    BEAM_WITHOUT_REFINEMENT = "beam_wt_refinement"
    BEAM_TUTORIAL = "beam_tutorial"
    BEAM_COMP = "beam_comp"
    BEAM_RANKED_IMAGE = "beam_ranked_image"
    BEAM_RANKED_BIGRAM = "beam_ranked_bigram"
    BEAM_PERPLEXITY = "perplexity"
    BEAM_SIM2IMAGE = "sim2image"
    BEAM_PERPLEXITY_SIM2IMAGE = "perplexity_image"
    BIGRAM_PROB = "bigram_prob"
    BIGRAM_PROB_IMAGE = "bigramprob_and_image"
    BIGRAM_PROB_COS = "bigramprob_and_cos"
