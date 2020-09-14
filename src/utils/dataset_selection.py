from utils.enums import Datasets, EvalDatasetType

PATH_RSICD = "src/data/RSICD/"
PATH_DATASETS_RSICD = PATH_RSICD + "datasets/"
PATH_UCM = "src/data/UCM/"
PATH_DATASETS_UCM = PATH_UCM + "datasets/"
PATH_DATASETS_RSICD_NEW_TRAIN_AND_VAL = PATH_RSICD + "datasets_new_train_and_val/"
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
    else:
        raise "Invalid dataset"


def get_train_and_val_paths(dataset):
    dataset_folder, dataset_jsons = get_dataset_paths(dataset)

    vocab_path = dataset_jsons + "vocab_info.json"
    train_json_path = dataset_jsons + "train.json"
    train_images_path = dataset_folder + "raw_dataset/RSICD_images/"
    val_json_path = dataset_jsons + "val.json"
    val_images_path = dataset_folder + "raw_dataset/RSICD_images/"

    return vocab_path, train_json_path, train_images_path, val_json_path, val_images_path


def get_test_path(args):
    dataset_folder, dataset_jsons = get_dataset_paths(args.dataset)

    if args.eval_dataset_type == EvalDatasetType.VAL.value:
        test_path = dataset_jsons + "val_coco_format.json"
        decoding_args = args.file_name + "_v_" + args.decodying_type + "_" + str(args.n_beam) + '_coco'

    elif args.eval_dataset_type == EvalDatasetType.TRAIN_AND_VAL.value:
        test_path = dataset_jsons + "train_and_val_coco_format.json"
        decoding_args = args.file_name + "_tv_" + args.decodying_type + "_" + str(args.n_beam) + '_coco'

    else:  # test set
        test_path = dataset_jsons + "test_coco_format.json"
        decoding_args = args.file_name + "_" + args.decodying_type + "_" + str(args.n_beam) + '_coco'

    return test_path, decoding_args
