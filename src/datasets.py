import torch
from torch.utils.data import Dataset
import json
from PIL import Image
from preprocess_data.tokens import convert_captions_to_Y
from create_data_files import get_dataset


class CaptionDataset(Dataset):

    def __init__(
        self,
        data_folder,
        images_folder,
        data_type,
        max_len,
        token_to_id,
        transform
    ):

        dataset = get_dataset(data_folder)

        self.images_names, captions_of_tokens = dataset[
            "images_names"], dataset["captions_tokens"]

        self.input_captions, self.target_captions, self.captions_lengths = convert_captions_to_Y(
            captions_of_tokens, max_len, token_to_id)

        # logging.info("len dataset of %s is %s",
        #              data_type, len(self.images_names))
        self.images_folder = images_folder
        self.dataset_size = len(self.images_names)
        self.transform = transform

    def __getitem__(self, i):
        image_name = self.images_folder + self.images_names[i]
        image = Image.open(image_name)

        image = self.transform(image)

        input_caption = self.input_captions[i]
        target_caption = self.target_captions[i]
        caption_lenght = self.captions_lengths[i]

        return image, torch.LongTensor(input_caption), torch.LongTensor(target_caption), torch.LongTensor([caption_lenght])

    def __len__(self):
        return self.dataset_size


class TrialDataset(Dataset):

    def __init__(
        self,
        data
    ):
        self.data = data

    def __getitem__(self, i):
        print("this is my data i", self.data[i])
        return self.data[i]

    def __len__(self):
        return len(self.data)
