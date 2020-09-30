from utils.enums import Datasets, EvalDatasetType

PATH_RSICD = "src/data/RSICD/"
PATH_DATASETS_RSICD = PATH_RSICD + "datasets/"
PATH_DATASETS_RSICD_NEW_TRAIN_AND_VAL = PATH_RSICD + "datasets_new_train_and_val/"
PATH_UCM = "src/data/UCM/"
PATH_DATASETS_UCM = PATH_UCM + "datasets/"
PATH_FLICKR8K = "src/data/FLICKR8k/"
PATH_DATASETS_FLICKR8K = PATH_FLICKR8K + "datasets/"
PATH_TRAINED_MODELS = "experiments/results/trained_models/"
PATH_EVALUATION_SENTENCES = "experiments/results/evaluation_sentences/"
PATH_EVALUATION_SCORES = "experiments/results/evaluation_scores/"


def get_dataset_paths(dataset):
    if dataset == Datasets.RSICD.value:
        print("entrei aqui no rscid")
        return PATH_RSICD, PATH_DATASETS_RSICD

    elif dataset == Datasets.RSICD_NEW_SPLITS_TRAIN_VAL.value:
        return PATH_RSICD, PATH_DATASETS_RSICD_NEW_TRAIN_AND_VAL

    elif dataset == Datasets.UCM.value:
        return PATH_UCM, PATH_DATASETS_UCM

    elif dataset == Datasets.FLICKR8K.value:
        return PATH_FLICKR8K, PATH_DATASETS_FLICKR8K

    else:
        raise "Invalid dataset"


def get_test_path(args, dataset_jsons):

    if args.eval_dataset_type == EvalDatasetType.VAL.value:
        test_path = dataset_jsons + "val_coco_format.json"
        decoding_args = args.file_name + "n_v_" + args.decodying_type + \
            "_" + str(args.n_beam) + "_" + str(args.min_len) + '_coco'

    else:  # test set
        test_path = dataset_jsons + "test_coco_format.json"
        decoding_args = args.file_name + "n_" + args.decodying_type + \
            "_" + str(args.n_beam) + "_" + str(args.min_len) + '_coco'

    return test_path, decoding_args
