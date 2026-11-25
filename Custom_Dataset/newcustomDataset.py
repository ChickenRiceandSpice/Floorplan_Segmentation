import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image as Image

# C:\Users\Yaash Rane\PycharmProjects\Image Detection\Result_evaluation\Cleaned_Images
# C:\Users\Yaash Rane\PycharmProjects\Image Detection\Result_evaluation\FloorPlans_dataset

class NewCustomDataset(Dataset):
    def __init__(self, csv_file, root_dir_raw, root_dir_clean, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir_raw = root_dir_raw
        self.root_dir_clean = root_dir_clean
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        raw_image_path = os.path.join(self.root_dir_raw, self.annotations.iloc[idx, 0])
        raw_image = Image.open(raw_image_path)

        cleaned_image_path = os.path.join(self.root_dir_clean, self.annotations.iloc[idx, 2])
        cleaned_image = Image.open(cleaned_image_path)

        if self.transform:
            raw_image = self.transform(raw_image)*255
            cleaned_image = self.transform(cleaned_image)*255

        return raw_image, cleaned_image