import json
from coco_caption.pycocotools.coco import COCO
from coco_caption.pycocoevalcap.eval import COCOEvalCap
from args_parser import get_args

from definitions import PATH_RSICD, PATH_DATASETS_RSICD, EVALUATION_SENTENCES, EVALUATION_SCORES


if __name__ == "__main__":

    args = get_args()
    print(args.__dict__)

    test_path = PATH_DATASETS_RSICD+"test_coco_format.json"

    sentences_path = EVALUATION_SENTENCES + \
        args.file_name + "_"+args.decodying_type + "_"+str(args.n_beam) + '_coco'

    coco = COCO(test_path)
    cocoRes = coco.loadRes(sentences_path+'.json')
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
    scores_path = EVALUATION_SCORES + \
        args.file_name + "_"+args.decodying_type + "_"+str(args.n_beam) + '_coco'  # str(self.args.__dict__)

    with open(scores_path+'.json', 'w+') as f:
        json.dump(predicted, f, indent=2)
