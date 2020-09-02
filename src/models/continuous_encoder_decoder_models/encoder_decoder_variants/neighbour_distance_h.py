import sys
sys.path.append('src/')

import torchvision
from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from models.basic_encoder_decoder_models.encoder_decoder import Encoder
from models.abtract_model import AbstractEncoderDecoderModel
import torch.nn.functional as F
from embeddings.embeddings import get_embedding_layer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from data_preprocessing.preprocess_tokens import OOV_TOKEN
from embeddings.embeddings import EmbeddingsType
from models.continuous_encoder_decoder_models.encoder_decoder import ContinuousEncoderDecoderModel
from embeddings.embeddings import EmbeddingsType
from torchvision import transforms
from definitions import PATH_DATASETS_RSICD, PATH_RSICD
from data_preprocessing.create_data_files import get_dataset
import cv2
from toolz import unique
from data_preprocessing.datasets import NeighbourDataset
from torch.utils.data import DataLoader
import faiss
from collections import defaultdict
from collections import Counter


class ContinuousNeighbourDHModel():

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_encoder_and_decoder()
        self.encoder.eval()

        print("sai do init")

    def _initialize_encoder_and_decoder(self):

        self.encoder = Encoder("efficient_net",  # EfficientNet
                               enable_fine_tuning=False)

        self.encoder = self.encoder.to(self.device)

    def distance_h(self, pairwise_similarity):
        term_1 = np.mean(np.min(1 - pairwise_similarity, 1))
        term_2 = np.mean(np.min(1 - pairwise_similarity, 0))
        dist = (term_1 + term_2) / 2
        return term_1, term_2, dist

    def setup_to_test(self):
        print("entrei aqui no setup")

        train_images = self.get_train()
        val_images = self.get_val()
        test_images = self.get_test()

        pairwise_sim = cosine_similarity(train_images, val_images)
        print("distance h", self.distance_h(pairwise_sim))

        #self.index, self.images_ids, self.dict_imageid_refs, self.counter_refs = self.create_index()

    def get_train(self):
        images_train = np.zeros((8734, 2076))
        for i in range(8734):
            images_train[i, :] = np.random.random((1, 2076))
        return images_train

    def get_val(self):
        images_train = np.zeros((1093, 2076))
        for i in range(1093):
            images_train[i, :] = np.random.random((1, 2076))
        return images_train

    def get_test(self):
        images_train = np.zeros((1093, 2076))
        for i in range(1093):
            images_train[i, :] = np.random.random((1, 2076))
        return images_train


if __name__ == "__main__":

    neigh_dh = ContinuousNeighbourDHModel()
    neigh_dh.setup_to_test()

    # def create_index(self):
    #     d = self.encoder.encoder_dim
    #     index = faiss.IndexFlatL2(d)
    #     images_ids = []

    #     train_dataset = get_dataset(PATH_DATASETS_RSICD + "train_coco_format.json")
    #     #train_dataset = get_dataset(PATH_DATASETS_RSICD + "val_coco_format.json")

    #     transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406],  # mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD
    #                              std=[0.229, 0.224, 0.225])
    #     ])

    #     for values in train_dataset["images"]:

    #         img_name = values["file_name"]
    #         image_id = values["id"]

    #         image_name = PATH_RSICD + \
    #             "raw_dataset/RSICD_images/" + img_name
    #         image = cv2.imread(image_name)
    #         image = transform(image)
    #         image = image.unsqueeze(0)

    #         images_ids.append(image_id)

    #         encoder_output = self.encoder(image)
    #         encoder_output = encoder_output.view(1, -1, encoder_output.size()[-1])
    #         mean_encoder_output = encoder_output.mean(dim=1)
    #         index.add(mean_encoder_output.numpy())

    #     dict_imageid_refs = defaultdict(list)
    #     all_captions = []
    #     for ref in train_dataset["annotations"]:
    #         image_id = ref["image_id"]
    #         caption = ref["caption"]
    #         all_captions.append(caption)
    #         dict_imageid_refs[image_id].append(caption)

    #     counter_refs = Counter(all_captions)

    #     return index, images_ids, dict_imageid_refs, counter_refs

# aqui por um main

    # sim = torch.cosine_similarity(mean_encoder_output, mean_encoder_neighbour)
    # print("sim scores", sim)
    # scores_similarity.append(sim.item())
    # image_names.append(image_name)

    # sorted_scores, sorted_indices = torch.sort(torch.tensor(scores_similarity),  descending=True, dim=-1)

    # best_image_index = sorted_indices[0]
    # best_image_name = image_names[best_image_index]
    # generated_sentence = train_dataset[best_image_name][0]  # there are 5 captions per image

    # captions = []
    # for batch_i, (image_name, caption) in enumerate(train_dataloader):

    #     image_path = PATH_RSICD + \
    #         "raw_dataset/RSICD_images/" + image_name
    #     image = Image.open(image_path)
    #     image = transform(image)
    #     image = image.unsqueeze(0)

    #     encoder_neighbour = self.encoder(image)
    #     encoder_neighbour = encoder_neighbour.view(1, -1, encoder_neighbour.size()[-1])
    #     mean_encoder_neighbour = encoder_neighbour.mean(dim=1)
    #     sim = torch.cosine_similarity(mean_encoder_output, mean_encoder_neighbour)
    #     print("sim scores", sim)
    #     scores_similarity.append(sim.item())
    #     captions.append(caption)

    # sorted_scores, sorted_indices = torch.sort(torch.tensor(scores_similarity), descending=True, dim=-1)

    # best_image_index = sorted_indices[0]
    # best_image_name = image_names[best_image_index]
    # generated_sentence = train_dataset[best_image_name][0]  # there are 5 captions per image

    # print("generated sentence", generated_sentence)

    # return generated_sentence  # input_caption

# image: vocab
