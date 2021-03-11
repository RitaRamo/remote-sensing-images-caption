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
#import inflect
from collections import Counter, OrderedDict, defaultdict
from utils.enums import Datasets
from definitions_datasets import get_dataset_paths
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import csv 
import operator

DATASET = "rsicd"

def glove2dict(glove_filename):
    with open(glove_filename, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=' ',quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:])))
                for line in reader}
    return embed


if __name__ == "__main__":

    # nlp = spacy.load("en_core_web_sm")
    # nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    # p = inflect.engine()

    dataset_folder, dataset_jsons = get_dataset_paths(DATASET)
    print("dataset folder", dataset_folder)

    train_dataset = get_dataset(dataset_jsons + "train.json")
    vocab_info = get_vocab_info(dataset_jsons + "vocab_info.json")
    vocab_size, token_to_id, id_to_token, max_len = vocab_info[
        "vocab_size"], vocab_info["token_to_id"], vocab_info["id_to_token"], vocab_info["max_len"]

    images_names, captions_of_tokens = train_dataset[
        "images_names"], train_dataset["captions_tokens"]


    # class MyIter:
    #     def __iter__(self):
    #         for i in range(len(captions_of_tokens)):
    #             yield captions_of_tokens[i]

    # print("len train", len(captions_of_tokens))
    # #print("My iter", next(iter(MyIter())))

    # #w2v_model = Word2Vec(size=300, window=3, min_count=2)

    # glove_input_file = 'src/embeddings/glove.6B/glove.6B.300d.txt'
    # word2vec_output_file = 'glove.6B.300d.txt.word2vec'
    # glove2word2vec(glove_input_file, word2vec_output_file)

    # w2v_model = KeyedVectors.load_word2vec_format(word2vec_output_file)
    # print(".wv.vocab", w2v_model.wv.vocab)
    # w2v_model.build_vocab(sentences=MyIter(), update=True)
    # total_examples = w2v_model.corpus_count
    # print("total exam", total_examples)
    # w2v_model.train(sentences=MyIter(), total_examples=total_examples, epochs=5)
    # w2v_model.save('src/embeddings/trained_embeddings.txt')
    # print(".wv.vocab", w2v_model.wv.vocab)



    token2id = {}
    vocab_freq_dict = {}
    # Populating vocab_freq_dict and token2id from my data.
    id_ = 0
    for words in captions_of_tokens:
        for word in words:
            if word not in vocab_freq_dict: vocab_freq_dict.update({word:0})
            vocab_freq_dict[word] += 1
            if word not in token2id:
                token2id.update({word:id_})
                id_ += 1

    #print("token to id", token2id)
    #print("vocab_freq_dict to id", vocab_freq_dict)
    #print(stop)

    # Populating vocab_freq_dict and token2id from glove vocab.
    embedding_name = "src/embeddings/glove.6B/glove.6B.300d.txt"
    max_id = max(token2id.items(), key=operator.itemgetter(1))[0]
    max_token_id = token2id[max_id]
    with open(embedding_name, encoding="utf8", errors='ignore') as f:
        for o in f:
            token, *vector = o.split(' ')
            token = str.lower(token)
            if len(o) <= 100: continue
            if token not in token2id:
                max_token_id += 1
                token2id.update({token:max_token_id})
                vocab_freq_dict.update({token:1})
                
    #with open("vocab_freq_dict","wb") as vocab_file: pickle.dump(vocab_freq_dict, vocab_file)
    #with open("token2id", "wb") as token2id_file: pickle.dump(token2id, token2id_file)
    print("\nnew token to id", token2id)
    print("\nnew vocab_freq_dict to id", vocab_freq_dict)
    print(stop)

    # converting vectors to keyedvectors format for gensim
    def load_vectors(token2id, path,  limit=None):
        embed_shape = (len(token2id), 300)
        freqs = np.zeros((len(token2id)), dtype='f')
        vectors = np.zeros(embed_shape, dtype='f')
        i = 0
        with open(path, encoding="utf8", errors='ignore') as f:
            for o in f:
                token, *vector = o.split(' ')
                token = str.lower(token)
                if len(o) <= 100: continue
                if limit is not None and i > limit: break
                vectors[token2id[token]] = np.array(vector, 'f')
                i += 1
        return vectors

    vectors = load_vectors(token2id, embedding_name)
    vec = KeyedVectors(300)
    vec.add(list(token2id.keys()), vectors, replace=True)
    vectors = None
    params = dict(min_count=1, workers=14, iter=6, size=300)
    model = Word2Vec(**params)
    model.build_vocab_from_freq(vocab_freq_dict)
    idxmap = np.array([token2id[w] for w in model.wv.index2entity])
    # Setting hidden weights(syn0 = between input/hidden layer and hidden/output layer) = your vectors arranged according to ids
    model.wv.vectors[:] = vec.vectors[idxmap]
    model.trainables.syn1neg[:] = vec.vectors[idxmap]
    # Train the model
    model.train(captions_of_tokens, total_examples=len(captions_of_tokens), epochs=5)
    output_path = 'src/embeddings/trained_embeddings.txt'
    model.save(output_path)

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