from create_data_files import get_vocab_info, get_dataset
from definitions import PATH_DATASETS_RSICD
from collections import defaultdict
import spacy
from spacy.tokens import Doc
from preprocess_data.tokens import WhitespaceTokenizer
import inflect
from collections import Counter
import torch
from collections import OrderedDict


VOCAB_SIZE = 600  # 512
dataset_path = "src/data/RSICD/datasets/pos_tagging_dataset"

if __name__ == "__main__":

    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    p = inflect.engine()

    train_dataset = get_dataset(PATH_DATASETS_RSICD+"train.json")
    vocab_info = get_vocab_info(PATH_DATASETS_RSICD+"vocab_info.json")
    vocab_size, token_to_id, id_to_token, max_len = vocab_info[
        "vocab_size"], vocab_info["token_to_id"], vocab_info["id_to_token"], vocab_info["max_len"]

    images_names, captions_of_tokens = train_dataset[
        "images_names"], train_dataset["captions_tokens"]

    image_categories = defaultdict(list)
    classes = []
    for i in range(len(images_names)):
        name = images_names[i]

        # append image class (obtained by the name ex: farmleand_111.jpeg)
        name_splited = name.split("_")
        if len(name_splited) > 1:
            img_class = name_splited[0]
            # image_categories[name].append(img_class)
            classes.append(img_class)
            print("class", img_class)
            if img_class == "sparseresidential":
                image_categories[name].append("sparse")
                image_categories[name].append("residential")
            elif img_class == "mediumresidential":
                image_categories[name].append("medium")
                image_categories[name].append("residential")
            elif img_class == "denseresidential":
                image_categories[name].append("dense")
                image_categories[name].append("residential")
            elif img_class == "storagetanks":
                image_categories[name].append("storage")
                image_categories[name].append("tanks")
            elif img_class == "railwaystation":
                image_categories[name].append("station")
                image_categories[name].append("railway")
            elif img_class == "baseballfield":
                image_categories[name].append("baseball")
                image_categories[name].append("field")

        # append words that are Nouns or Adjectives (converted to singular)
        caption = captions_of_tokens[i]
        tokens_without_special_tokens = caption[1:-1]
        sentence = " ".join([token for token in tokens_without_special_tokens if token != ""])
        print("i this is sentence", i, sentence)

        doc = nlp(sentence) if sentence != "" else []

        for spacy_token in doc:
            pos = spacy_token.pos_
            if pos == "NOUN" or pos == "ADJ":
                word = spacy_token.text
                # word_converted_to_singular = p.singular_noun(word)
                # if word_converted_to_singular:
                #     image_categories[name].append(word_converted_to_singular)
                # else:
                image_categories[name].append(word)

        #print("image category",  image_categories[name])
    print("all class", classes)

    # Each image has list of words/categories from all the captions
    lists_categories = list(image_categories.values())
    all_words = [item for sublist in lists_categories for item in sublist]

    vocab_words, counts = list(zip(*Counter(all_words).most_common(VOCAB_SIZE)))
    classes_to_id = OrderedDict([(value, index)
                                 for index, value in enumerate(vocab_words)])

    id_to_classes = OrderedDict([(index, value)
                                 for index, value in enumerate(vocab_words)])

    # filter categories by words of vocab, as well as remove repeated words within same image
    image_vocabcategories = {}
    for image_name, list_of_words in image_categories.items():
        categories_belonging_to_vocab = set([word for word in list_of_words if word in vocab_words])

        if len(categories_belonging_to_vocab) > 0:
            image_vocabcategories[image_name] = categories_belonging_to_vocab
        else:
            raise Exception("there is a image without any category associated", image_name)

    lists_final = list(image_vocabcategories.values())
    final_words = [item for sublist in lists_final for item in sublist]
    print("counter", Counter(final_words))
    print("len final", len(Counter(final_words)))

    classid_to_wordid = {}
    list_wordid = []
    for classe, id in classes_to_id.items():
        classid_to_wordid[id] = token_to_id[classe]
        list_wordid.append(token_to_id[classe])

    state = {
        "classification_dataset": image_vocabcategories,
        "classes_to_id": classes_to_id,
        "id_to_classes": id_to_classes,
        "classid_to_wordid": classid_to_wordid,
        "list_wordid": list_wordid
    }

    torch.save(state, "src/data/RSICD/datasets/classification_dataset_600")
