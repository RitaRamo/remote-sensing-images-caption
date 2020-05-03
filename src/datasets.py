import torch
from torch.utils.data import Dataset
from PIL import Image
from preprocess_data.tokens import convert_captions_to_Y
from preprocess_data.images import augment_image_with_color, augment_image_with_rotations_and_flips, augment_image
from create_data_files import get_dataset
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import imageio
from torchvision import transforms


class CaptionDataset(Dataset):

    def __init__(
        self,
        data_folder,
        images_folder,
        data_type,
        max_len,
        token_to_id,
        augmentation=False
    ):

        dataset = get_dataset(data_folder)

        self.images_names, captions_of_tokens = dataset[
            "images_names"], dataset["captions_tokens"]

        self.input_captions, self.captions_lengths = convert_captions_to_Y(
            captions_of_tokens, max_len, token_to_id)

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
        #image = Image.open(image_name)
        #image = self.transform(image)

        image = cv2.imread(image_name)
        image = self.get_transformed_image(image)
        #image = self.transform(image)

        input_caption = self.input_captions[i]
        caption_lenght = self.captions_lengths[i]

        return image, torch.LongTensor(input_caption), torch.LongTensor([caption_lenght])

    def __len__(self):
        return self.dataset_size


# class POSCaptionDataset(CaptionDataset):

#     def __init__(
#         self,
#         data_folder,
#         images_folder,
#         data_type,
#         max_len,
#         token_to_id,
#         augmentation=False
#     ):

#     super().__init__(data_folder,images_folder,data_type,max_len,token_to_id,augmentation)


#     def __getitem__(self, i):
#         image_name = self.images_folder + self.images_names[i]
#         image = cv2.imread(image_name)
#         image = self.get_transformed_image(image)

#         input_caption = self.input_captions[i]
#         #https://spacy.io/api/annotation#pos-tagging
#         #tens de garantir q fazes o mesmo split


#         caption_lenght = self.captions_lengths[i]


#         return image, torch.LongTensor(input_caption), torch.LongTensor([caption_lenght])

# class TestDataset(Dataset):

#     def __init__(
#         self,
#         data_folder,
#         images_folder,
#         data_type,
#         max_len,
#         token_to_id,
#         transform
#     ):

#         self.dataset = get_dataset(data_folder)

#         dataset_items = test_dataset.items

#         self.data=iter(test_dataset.items):

#         self.input_captions, self.target_captions, self.captions_lengths = convert_captions_to_Y(
#             captions_of_tokens, max_len, token_to_id)

#         # logging.info("len dataset of %s is %s",
#         #              data_type, len(self.images_names))
#         self.images_folder = images_folder
#         self.dataset_size = len(self.images_names)
#         self.transform = transform

#     def __getitem__(self, i):

#         dataset.items
#         image_name = self.images_folder + self.images_names[i]
#         image = Image.open(image_name)

#         image = self.transform(image)

#         input_caption = self.input_captions[i]
#         target_caption = self.target_captions[i]
#         caption_lenght = self.captions_lengths[i]

#         return image, torch.LongTensor(input_caption), torch.LongTensor(target_caption), torch.LongTensor([caption_lenght])

#     def __len__(self):
#         print("entrei aqui no LEN", self.dataset_size)
#         return self.dataset_size


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
