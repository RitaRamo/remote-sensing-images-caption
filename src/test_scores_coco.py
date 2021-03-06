import json
from coco_caption.pycocotools.coco import COCO
from coco_caption.pycocoevalcap.eval import COCOEvalCap
from args_parser import get_args

#from utils.enums import EvalDatasetType
from definitions_datasets import get_dataset_paths, get_test_path, PATH_EVALUATION_SENTENCES, PATH_EVALUATION_SCORES


if __name__ == "__main__":

    args = get_args()
    print(args.__dict__)

    # Choose dataset to evaluate the model:
    # if args.eval_dataset_type == EvalDatasetType.VAL.value:
    #     test_path = PATH_DATASETS_RSICD_NEW_TRAIN_AND_VAL + "val_coco_format.json"
    #     decoding_args = args.file_name + "_v_" + args.decodying_type + "_" + str(args.n_beam) + '_coco'

    # elif args.eval_dataset_type == EvalDatasetType.TRAIN_AND_VAL.value:
    #     test_path = PATH_DATASETS_RSICD_NEW_TRAIN_AND_VAL + "train_and_val_coco_format.json"
    #     decoding_args = args.file_name + "_tv_" + args.decodying_type + "_" + str(args.n_beam) + '_coco'

    # else:  # test set
    #     decoding_args = args.file_name + "_" + args.decodying_type + "_" + str(args.n_beam) + '_coco'
    #     test_path = PATH_DATASETS_RSICD_NEW_TRAIN_AND_VAL + "test_coco_format.json"
    dataset_folder, dataset_jsons = get_dataset_paths(args.dataset)
    test_path, decoding_args = get_test_path(args, dataset_jsons)
    generated_sentences_path = PATH_EVALUATION_SENTENCES + decoding_args

    coco = COCO(test_path)
    cocoRes = coco.loadRes(generated_sentences_path + '.json')
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params["image_id"] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # save each image score and the avg score to a dict
    predicted = {"args": args.__dict__}
    individual_scores = [eva for eva in cocoEval.evalImgs]
    for i in range(len(individual_scores)):
        predicted[individual_scores[i]["image_id"]] = individual_scores[i]
    predicted["avg_metrics"] = cocoEval.eval

    # save scores dict to a json
    scores_path = PATH_EVALUATION_SCORES + decoding_args
    with open(scores_path + '.json', 'w+') as f:
        json.dump(predicted, f, indent=2)
