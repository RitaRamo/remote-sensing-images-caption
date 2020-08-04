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


class ContinuousNeighbourModel(ContinuousEncoderDecoderModel):

    def __init__(self,
                 args,
                 vocab_size,
                 token_to_id,
                 id_to_token,
                 max_len,
                 device
                 ):
        super().__init__(args, vocab_size, token_to_id, id_to_token, max_len, device)

    def _initialize_encoder_and_decoder(self):

        self.encoder = Encoder(self.args.image_model_type,
                               enable_fine_tuning=self.args.fine_tune_encoder)

        self.encoder = self.encoder.to(self.device)

    def setup_to_test(self):
        self._initialize_encoder_and_decoder()
        self.encoder.eval()

        print("using faiss to create index")
        self.index, self.images_ids, self.dict_imageid_refs = self.create_index()

    def create_index(self):
        d = self.encoder.encoder_dim
        index = faiss.IndexFlatL2(d)
        images_ids = []

        train_dataset = get_dataset(PATH_DATASETS_RSICD + "train_coco_format.json")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD
                                 std=[0.229, 0.224, 0.225])
        ])

        for values in train_dataset["images"]:

            img_name = values["file_name"]
            image_id = values["id"]

            image_name = PATH_RSICD + \
                "raw_dataset/RSICD_images/" + img_name
            image = cv2.imread(image_name)
            image = transform(image)
            image = image.unsqueeze(0)

            images_ids.append(image_id)

            encoder_output = self.encoder(image)
            encoder_output = encoder_output.view(1, -1, encoder_output.size()[-1])
            mean_encoder_output = encoder_output.mean(dim=1)
            index.add(mean_encoder_output.numpy())

        dict_imageid_refs = defaultdict(list)
        for ref in train_dataset["annotations"]:
            image_id = ref["image_id"]
            caption = ref["caption"]
            dict_imageid_refs[image_id].append(caption)

        return index, images_ids, dict_imageid_refs

    def inference_with_greedy(self, image, n_solutions=0):

        with torch.no_grad():  # no need to track history

            encoder_output = self.encoder(image)
            encoder_output = encoder_output.view(
                1, -1, encoder_output.size()[-1])
            mean_encoder_output = encoder_output.mean(dim=1)

            D, I = self.index.search(mean_encoder_output.numpy(), 1)
            nearest_img = self.images_ids[I[0][0]]  # pick first batch -> then pick first neighbour
            generated_sentence = self.dict_imageid_refs[nearest_img][0]  # pick first ref

            print("nearest caption", generated_sentence)

        return generated_sentence

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
