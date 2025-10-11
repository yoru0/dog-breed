import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

breeds_df = pd.read_csv("./dog-breed-identification/labels.csv")
breeds_df.head()


def barw(ax):
    for p in ax.patches:
        val = p.get_width()
        x = p.get_x() + p.get_width()
        y = p.get_y() + p.get_height() / 2
        ax.annotate(round(val, 2), (x, y))


plt.figure(figsize=(15, 30))
ax0 = sns.countplot(y=breeds_df["breed"], order=breeds_df["breed"].value_counts().index)
barw(ax0)
plt.show()

dog_breeds = sorted(list(set(breeds_df["breed"])))
n_breeds = len(dog_breeds)
print(n_breeds)
dog_breeds[:10]

breed_to_num = dict(zip(dog_breeds, range(n_breeds)))
breed_to_num

num_to_breed = {num: breed for breed, num in breed_to_num.items()}
num_to_breed


class DogDataset(Dataset):
    def __init__(
        self,
        csv_path,
        file_path,
        mode="train",
        transform=None,
        valid_ratio=0.2,
        resize_height=224,
        resize_width=224,
    ):
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.file_path = file_path
        self.mode = mode
        self.transform = transform

        if csv_path != None:
            self.data_info = pd.read_csv(csv_path)
            self.data_len = len(self.data_info.index)
            self.train_len = int(self.data_len * (1 - valid_ratio))

        if mode == "train":
            self.train_image = np.asarray(self.data_info.iloc[0 : self.train_len, 0])
            self.train_breed = np.asarray(self.data_info.iloc[0 : self.train_len, 1])
            self.image_arr = self.train_image
            self.breed_arr = self.train_breed
        elif mode == "valid":
            self.valid_image = np.asarray(self.data_info.iloc[self.train_len :, 0])
            self.valid_breed = np.asarray(self.data_info.iloc[self.train_len :, 1])
            self.image_arr = self.valid_image
            self.breed_arr = self.valid_breed
        elif mode == "test":
            self.image_arr = [f.split(".")[0] for f in os.listdir(file_path)]

        self.real_len = len(self.image_arr)
        print(
            "Finished reading the {} set of Leaves Dataset ({} samples found)".format(
                mode, self.real_len
            )
        )

    def __getitem__(self, index):
        single_image_name = self.image_arr[index]

        img_as_img = Image.open(self.file_path + single_image_name + ".jpg")
        img_as_img = self.transform(img_as_img)

        if self.mode == "test":
            return img_as_img, single_image_name
        else:
            breed = self.breed_arr[index]
            number_breed = breed_to_num[breed]
            return img_as_img, number_breed

    def __len__(self):
        return self.real_len


train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            224, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

train_path = "/kaggle/input/dog-breed-identification/labels.csv"
train_img_path = "/kaggle/input/dog-breed-identification/train/"
test_img_path = "/kaggle/input/dog-breed-identification/test/"

train_dataset = DogDataset(
    train_path, train_img_path, mode="train", transform=train_transform
)
valid_dataset = DogDataset(
    train_path, train_img_path, mode="valid", transform=test_transform
)
test_dataset = DogDataset(None, test_img_path, mode="test", transform=test_transform)
print(train_dataset)
print(valid_dataset)
print(test_dataset)

batch_size = 128
train_iter = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
)

valid_iter = DataLoader(
    dataset=valid_dataset, batch_size=batch_size, shuffle=True, drop_last=False
)

test_iter = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


device = get_device()
print(device)


def get_net(device):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet50(
        weights="ResNet50_Weights.DEFAULT"
    )
    finetune_net.output_new = nn.Sequential(
        nn.Linear(1000, 256), nn.ReLU(), nn.Linear(256, 120)
    )
    finetune_net = finetune_net.to(device)
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net


learning_rate = 1e-4
weight_decay = 1e-3
num_epoch = 50
model_path = "/kaggle/working/pre_res_model.ckpt"

