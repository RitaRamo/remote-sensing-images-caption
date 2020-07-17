import torch
from torch.utils.data import Dataset
from PIL import Image
from data_preprocessing.preprocess_tokens import convert_captions_to_Y, convert_captions_to_Y_and_POS
from data_preprocessing.preprocess_images import augment_image_with_color, augment_image_with_rotations_and_flips, augment_image
from data_preprocessing.create_data_files import get_dataset
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import imageio
from torchvision import transforms
import logging
import os


class CaptionDataset(Dataset):

    def __init__(
        self,
        data_folder,
        images_folder,
        max_len,
        token_to_id,
        augmentation=False
    ):

        self._init_caption(data_folder, max_len, token_to_id)
        self._init_images(images_folder, augmentation)

    def _init_caption(self, data_folder, max_len, token_to_id):
        dataset = get_dataset(data_folder)

        self.images_names, captions_of_tokens = dataset[
            "images_names"], dataset["captions_tokens"]

        self.input_captions, self.captions_lengths = convert_captions_to_Y(
            captions_of_tokens, max_len, token_to_id)

    def _init_images(self, images_folder, augmentation):
        self.images_folder = images_folder
        self.dataset_size = len(self.images_names)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD
                                 std=[0.229, 0.224, 0.225])
        ])

        if augmentation:
            self.get_transformed_image = self.get_image_augmented

        else:
            self.get_transformed_image = self.get_torch_image

    def get_image_augmented(self, image):
        image = augment_image()(image=image)["image"]
        return self.get_torch_image(image)

    def get_torch_image(self, image):
        return self.transform(image)

    def __getitem__(self, i):
        image_name = self.images_folder + self.images_names[i]
        # image = Image.open(image_name)
        # image = self.transform(image)

        image = cv2.imread(image_name)
        image = self.get_transformed_image(image)
        # image = self.transform(image)

        input_caption = self.input_captions[i]
        caption_lenght = self.captions_lengths[i]

        return image, torch.LongTensor(input_caption), torch.LongTensor([caption_lenght])
        # TODO: CHANGE torch.long

    def __len__(self):
        return self.dataset_size


class POSCaptionDataset(CaptionDataset):

    def __init__(
        self,
        data_folder,
        images_folder,
        max_len,
        token_to_id,
        augmentation=False
    ):
        print("entrei aqui")
        super().__init__(data_folder,
                         images_folder,
                         max_len,
                         token_to_id,
                         augmentation
                         )

    def _init_caption(self, data_folder, max_len, token_to_id):
        dataset = get_dataset(data_folder)

        self.images_names, captions_of_tokens = dataset[
            "images_names"], dataset["captions_tokens"]

        dataset_path = "src/data/RSICD/datasets/pos_tagging_dataset"

        if os.path.exists(dataset_path):
            loaded_dataset = torch.load(dataset_path)
            self.input_captions = loaded_dataset["input_captions"]
            self.captions_lengths = loaded_dataset["captions_lengths"]

        else:
            logging.info("loading caption and pos tagging")
            self.input_captions, self.captions_lengths = convert_captions_to_Y_and_POS(
                captions_of_tokens, max_len, token_to_id)

            state = {
                "input_captions": self.input_captions,
                "captions_lengths": self.captions_lengths
            }

            torch.save(state, dataset_path)

    def __getitem__(self, i):
        image_name = self.images_folder + self.images_names[i]
        # image = Image.open(image_name)
        # image = self.transform(image)

        image = cv2.imread(image_name)
        image = self.get_transformed_image(image)
        # image = self.transform(image)

        input_caption = self.input_captions[i]
        caption_lenght = self.captions_lengths[i]

        return image, input_caption, torch.LongTensor([caption_lenght])


class ClassificationDataset(CaptionDataset):
    def __init__(
        self,
        data,
        images_folder,
        classes_to_id,
        augmentation=True
    ):
        self.images_names, categories = zip(*(data.items()))
        super()._init_images(images_folder, augmentation)
        self._init_categories(categories, classes_to_id)

    def _init_categories(self, categories, classes_to_id):
        # categories=items()
        vocab_size = len(classes_to_id)
        # tens de faze
        self.categories_tensor = torch.zeros(self.dataset_size, vocab_size)

        for i in range(len(categories)):

            categories_to_integer = [classes_to_id[category] for category in categories[i]]

            self.categories_tensor[i, [categories_to_integer]] = 1

    def __getitem__(self, i):
        image_name = self.images_folder + self.images_names[i]
        image = cv2.imread(image_name)
        image = self.get_transformed_image(image)

        classes = self.categories_tensor[i]

        return image, classes


class ClassificationContinuousDataset(CaptionDataset):

    def __init__(
        self,
        images_names,
        captions_of_tokens,
        images_folder,
        max_len,
        token_to_id,
        augmentation=False
    ):

        self._init_caption(images_names, captions_of_tokens, max_len, token_to_id)
        self._init_images(images_folder, augmentation)

    def _init_caption(self, images_names, captions_of_tokens, max_len, token_to_id):

        self.images_names = images_names

        self.input_captions, self.captions_lengths = convert_captions_to_Y(
            captions_of_tokens, max_len, token_to_id)

    def __getitem__(self, i):
        image_name, caption, caption_len = super().__getitem__(i)

        return image_name, caption


class NeighbourDataset(Dataset):

    def __init__(
        self,
        data_folder,
        max_len,
        token_to_id,
        augmentation=False
    ):

        self.dataset = list(get_dataset(data_folder).items())
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD
                                 std=[0.229, 0.224, 0.225])
        ])
        self.get_transformed_image = self.get_torch_image

    def get_torch_image(self, image):
        return self.transform(image)

    def __getitem__(self, i):

        current_item = self.dataset[i]
        image_name = self.images_folder + current_item[0]  # [0]-> image
        # image = Image.open(image_name)
        # image = self.transform(image)

        image = cv2.imread(image_name)
        image = self.get_transformed_image(image)
        # image = self.transform(image)

        caption = current_item[1][0]  # [1]->sentences [0]-> first sentence of all 5 sentences

        return image, caption

    def __len__(self):
        return self.dataset_size


class TrialDataset(Dataset):

    def __init__(
        self,
        data
    ):
        self.data = data
        print("this is the data", data)
        self.data_size = len(self.data)
        print("this is the size of data", self.data_size)

    def __getitem__(self, i):
        print("this is my data i", self.data[i])
        return self.data[i]

    def __len__(self):
        print("what is data size", self.data_size)
        return self.data_size
