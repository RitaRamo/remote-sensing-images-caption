import sys
sys.path.append('src/')
# sys.path.append('src/data_preprocessing')

# sys.path.append('../crea')


from data_preprocessing.preprocess_tokens import WhitespaceTokenizer
from definitions_datasets import PATH_DATASETS_RSICD
from data_preprocessing.create_data_files import get_dataset, get_vocab_info
from spacy.tokens import Doc
import torch
import spacy
import inflect
from collections import Counter, OrderedDict, defaultdict
from utils.enums import Datasets
from definitions_datasets import get_dataset_paths

DATASET = "rsicd"

if __name__ == "__main__":

    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    p = inflect.engine()

    dataset_folder, dataset_jsons = get_dataset_paths(DATASET)
    print("dataset folder", dataset_folder)

    train_dataset = get_dataset(dataset_jsons + "train.json")
    vocab_info = get_vocab_info(dataset_jsons + "vocab_info.json")
    vocab_size, token_to_id, id_to_token, max_len = vocab_info[
        "vocab_size"], vocab_info["token_to_id"], vocab_info["id_to_token"], vocab_info["max_len"]

    images_names, captions_of_tokens = train_dataset[
        "images_names"], train_dataset["captions_tokens"]

    image_caption = defaultdict(list)
    classes = []

    for i in range(len(images_names)):
        name = images_names[i]

        # append words that are Nouns or Adjectives (converted to singular)
        caption = captions_of_tokens[i]
        tokens_without_special_tokens = caption[1:-1]
        image_caption[name] = [token_to_id[token] for token in tokens_without_special_tokens]

    state = {
        "classification_dataset": image_caption,  # image to word ids of caption
    }

    if DATASET == Datasets.RSICD.value:
        torch.save(state, dataset_jsons + "classification_dataset_rsicd_caption")

    elif DATASET == Datasets.UCM.value:
        torch.save(state, dataset_jsons + "classification_dataset_ucm_caption")
    elif DATASET == Datasets.FLICKR8K.value:
        torch.save(state, dataset_jsons + "classification_dataset_flickr8k_caption")
    else:
        raise Exception("unknown dataset to save")