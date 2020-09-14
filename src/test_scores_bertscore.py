import json

from args_parser import get_args

from definitions import PATH_RSICD, PATH_DATASETS_RSICD_NEW_TRAIN_AND_VAL, PATH_EVALUATION_SENTENCES, PATH_EVALUATION_SCORES
from collections import defaultdict
from bert_score import BERTScorer
from utils.enums import EvalDatasetType


if __name__ == "__main__":
    # add bert_scores to coco metrics
    scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    args = get_args()
    print(args.__dict__)

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

    with open(test_path) as json_file:
        test = json.load(json_file)

    dict_imageid_refs = defaultdict(list)
    for ref in test["annotations"]:
        image_id = ref["image_id"]
        caption = ref["caption"]
        dict_imageid_refs[image_id].append(caption)

    # get previous score of coco metrics (bleu,meteor,etc) to append bert_score
    scores_path = PATH_EVALUATION_SCORES + decoding_args

    with open(scores_path + '.json') as json_file:
        scores = json.load(json_file)

    # get previous generated sentences to calculate bertscore according to refs
    generated_sentences_path = PATH_EVALUATION_SENTENCES + decoding_args
    with open(generated_sentences_path + ".json") as json_file:
        generated_sentences = json.load(json_file)

    total_precision = 0.0
    total_recall = 0.0
    total_f = 0.0
    for dict_image_and_caption in generated_sentences:
        image_id = dict_image_and_caption["image_id"]
        caption = [dict_image_and_caption["caption"]]
        references = [dict_imageid_refs[image_id]]

        P_mul, R_mul, F_mul = scorer.score(caption, references)
        precision = P_mul[0].item()
        recall = R_mul[0].item()
        f_measure = F_mul[0].item()

        total_precision += precision
        total_recall += recall
        total_f += f_measure

        # calculate bert_score
        key_image_id = str(image_id)
        scores[str(key_image_id)]["BertScore_P"] = precision
        scores[key_image_id]["BertScore_R"] = recall
        scores[key_image_id]["BertScore_F"] = f_measure
        print("\ncaption and score", caption, f_measure)

    n_captions = len(generated_sentences)
    scores["avg_metrics"]["BertScore_P"] = total_precision / n_captions
    scores["avg_metrics"]["BertScore_R"] = total_recall / n_captions
    scores["avg_metrics"]["BertScore_F"] = total_f / n_captions

    decoding_args = args.file_name + "_" + args.decodying_type + "_" + str(args.n_beam) + '_coco'

    # save scores dict to a json
    with open(scores_path + '.json', 'w+') as f:
        json.dump(scores, f, indent=2)
