from create_data_files import get_vocab_info, get_dataset, PATH_DATASETS_RSICD
from toolz import unique
import torch
from models.continuous_encoder_decoder_models.encoder_decoder_variants.bert import BertDecoder
from preprocess_data.tokens import START_TOKEN, END_TOKEN
from transformers import BertTokenizer, BertModel
from collections import OrderedDict

if __name__ == "__main__":
    train_dataset = get_dataset(PATH_DATASETS_RSICD+"train.json")
    vocab_info = get_vocab_info(PATH_DATASETS_RSICD+"vocab_info.json")
    vocab_size, token_to_id, id_to_token, max_len = vocab_info[
        "vocab_size"], vocab_info["token_to_id"], vocab_info["id_to_token"], vocab_info["max_len"]

    images_names, captions_of_tokens = train_dataset[
        "images_names"], train_dataset["captions_tokens"]

    all_sentences = [" ".join(sentence).replace(START_TOKEN, " [CLS] ").replace(
        END_TOKEN, " [SEP] ") for sentence in captions_of_tokens]
    unique_sentences = set(all_sentences)

    bert_tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()

    bert_decoder = BertDecoder(
        bert_tokenizer=bert_tokenizer,
        bert_model=bert_model,
        decoder_dim=512,
        vocab_size=vocab_size,
        token_to_id=token_to_id,
        id_to_token=id_to_token
    )

    pretrained_embeddings_matrix = torch.zeros((vocab_size, 768))

    for word, id in token_to_id.items():
        if word == START_TOKEN:
            word = "[CLS]"
        elif word == END_TOKEN:
            word = "[SEP]"

        word_and_not_suboword = " "+word+" "
        sentences_contain_word = [
            sentence for sentence in unique_sentences if word_and_not_suboword in sentence]
        print("word and n_sent", word, len(sentences_contain_word))

        n_embeddings = max(1, len(sentences_contain_word))
        word_embeddings = torch.zeros((n_embeddings, 768))
        embedding_i = 0

        for sentence in sentences_contain_word:
            word_index = sentence.index(word_and_not_suboword)
            sentence_until_word = sentence[:word_index +
                                           len(word_and_not_suboword)]
            contextual_embedding = bert_decoder.get_contextualize_embedding(
                sentence_until_word)
            word_embeddings[embedding_i, :] = contextual_embedding
            embedding_i += 1

        pretrained_embeddings_matrix[id, :] = torch.mean(
            word_embeddings, dim=0)

    state = {'pretrained_embeddings_matrix': pretrained_embeddings_matrix}

    torch.save(state, "src/embeddings/bert/bert_matrix_final.pth.tar")
