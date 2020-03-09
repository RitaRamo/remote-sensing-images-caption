import logging
from args_parser import get_args
from create_data_files import PATH_RSICD, PATH_DATASETS_RSICD, get_vocab_info, get_dataset
from models.attention.attention_model import AttentionModel
from models.basic_model import BasicModel
#from models.attention.attention_model_old import AttentionOldModel
from models.continuous.basic_continuous import BasicContinuousModel
from models.continuous.attention_continuous import AttentionContinuousModel

import torch


import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = '0'

if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)

    device = torch.device("cpu")

    args = get_args()
    logging.info(args.__dict__)

    vocab_info = get_vocab_info(PATH_DATASETS_RSICD+"vocab_info.json")
    vocab_size, token_to_id, id_to_token, max_len = vocab_info[
        "vocab_size"], vocab_info["token_to_id"], vocab_info["id_to_token"], vocab_info["max_len"]
    logging.info("vocab size %s", vocab_size)

    test_dataset = get_dataset(PATH_DATASETS_RSICD+"test.json")

    model_class = globals()[args.model_class_str]
    model = model_class(
        args, vocab_size, token_to_id, id_to_token, max_len, device)
    model.setup_to_test()
    scores = model.test(test_dataset)
    #scores = model.evaluate(1, test_dataset)
    model.save_scores(scores)
