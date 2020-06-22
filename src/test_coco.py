import json
from coco_caption.pycocotools.coco import COCO
from coco_caption.pycocoevalcap.eval import COCOEvalCap


if __name__ == "__main__":
    device = torch.device("cpu")

    args = get_args()
    print(args.__dict__)

    vocab_info = get_vocab_info(PATH_DATASETS_RSICD+"vocab_info.json")
    vocab_size, token_to_id, id_to_token, max_len = vocab_info[
        "vocab_size"], vocab_info["token_to_id"], vocab_info["id_to_token"], vocab_info["max_len"]
    print("vocab size", vocab_size)

    test_dataset = get_dataset(PATH_DATASETS_RSICD+"test_coco_format.json")

    model_class = globals()[args.model_class_str]
    model = model_class(
        args, vocab_size, token_to_id, id_to_token, max_len, device)
    model.setup_to_test()

    sentences_path = model.MODEL_DIRECTORY + \
        'evaluation_sentences/' + \
        args.file_name + "_"+args.decodying_type + "_"+str(args.n_beam) + '_coco'  # str(self.args.__dict__)

    with open(sentences_path+'.json', 'w+') as f:
        json.dump(list_hipotheses, f, indent=2)

    coco = COCO(test_dataset)
    cocoRes = coco.loadRes(sentences_path+'.json')

    cocoEval = COCOEvalCap(coco, cocoRes)

    cocoEval.params["image_id"] = cocoRes.getImgIds()
    cocoEval.evaluate()

    predicted = {"args": [args.__dict__]}
    avg_score = cocoEval.eval.items()
    individual_scores = [eva for eva in cocoEval.evalImgs]
    for i in range(len(individual_scores)):
        predicted[individual_scores[i]["image_id"]] = individual_scores[i]
    predicted["avg_metrics"] = avg_score

    model.save_scores(args.decodying_type, args.n_beam, predicted, True)

    scores_path = model.MODEL_DIRECTORY + \
        'evaluation_scores/' + \
        args.file_name + "_"+args.decodying_type + "_"+str(args.n_beam) + '_coco'  # str(self.args.__dict__)

    with open(scores_path+'.json', 'w+') as f:
        json.dump(predicted, f, indent=2)
