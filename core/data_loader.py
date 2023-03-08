import glob
import os
from typing import List

import numpy as np
import pandas as pd
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchsampler import ImbalancedDatasetSampler
from torchvision.transforms import transforms as trns


def create_data(data_dir: os.pardir, save_file: str) -> pd.DataFrame:
    """
    It takes in the data directory and the name of the file to save the data to, and returns a pandas
    dataframe with the image paths and labels

    :param data_dir: The directory where the data is stored
    :type data_dir: os.pardir
    :param save_file: The name of the file to save the data to
    :type save_file: str
    :return: A dataframe with the image and label columns.
    """
    normal_cases_dir = os.path.join(data_dir, "NORMAL")
    pneumonia_cases_dir = os.path.join(data_dir, "PNEUMONIA")

    normal_cases = glob.glob(f"{normal_cases_dir}/*.jpeg")
    pneumonia_cases = glob.glob(f"{pneumonia_cases_dir}/*.jpeg")

    data = []

    for img in normal_cases:
        data.append((img, 0))

    for img in pneumonia_cases:
        data.append((img, 1))

    data = pd.DataFrame(data, columns=["image", "label"], index=None)

    data = data.sample(frac=1.0).reset_index(drop=True)

    os.makedirs("data_chestxray", exist_ok=True)
    data.to_csv(os.path.join("data_chestxray/", save_file), index=False)
    return data


def concat_dataset(
    train_dir: str = "chest_xray/train",
    test_dir: str = "chest_xray/test",
    val_dir: str = "chest_xray/val",
) -> pd.DataFrame:
    """
    It takes in the train and validation directory and creates a dataframe for each of them. Then it
    concatenates the two dataframes and returns the concatenated dataframe

    :param train_dir: The directory where the training data is stored, defaults to chest_xray/train
    :type train_dir: str (optional)
    :param val_dir: The directory where the validation data is stored, defaults to chest_xray/val
    :type val_dir: str (optional)
    :return: A dataframe with the image paths and labels for the train and validation sets.
    """
    data_train = create_data(data_dir=train_dir, save_file="train.csv")
    if test_dir:
        data_test = create_data(data_dir=test_dir, save_file="test.csv")
    else:
        data_test = None
    if val_dir:
        data_validation = create_data(data_dir=val_dir, save_file="val.csv")
    else:
        data_validation = None

    data_merge = (
        pd.concat([data_train, data_validation, data_test], axis=0)
        .reset_index()
        .drop(["index"], axis=1)
    )

    return data_merge


def data_augmentations(img_size: float = 224):
    """
    It takes an image, resizes it to 224x224, flips it horizontally, converts it to a tensor, and
    normalizes it

    :param img_size: The size of the image to be resized to, defaults to 224
    :type img_size: float (optional)
    :return: the train_transforms, val_transforms, and test_transforms.
    """

    train_transforms = trns.Compose(
        [
            trns.CenterCrop(img_size),
            trns.RandomHorizontalFlip(),
            trns.RandomVerticalFlip(),
            trns.ToTensor(),
            trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transforms = trns.Compose(
        [
            trns.Resize(size=img_size),
            trns.ToTensor(),
            trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_transforms = trns.Compose(
        [
            trns.Resize(size=img_size),
            trns.ToTensor(),
            trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transforms, val_transforms, test_transforms


class ChestXrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms: List = None) -> None:
        """
        This function takes in a dataframe and a list of transforms and assigns the dataframe to the
        class variable df and the list of transforms to the class variable transforms

        :param df: The dataframe containing the data
        :type df: pd.DataFrame
        :param transforms: List = None
        :type transforms: List
        """
        self.df = df
        self.transforms = transforms
        self.labels = self.get_labels()

    def get_labels(self):
        labels = self.df["label"].values
        return labels

    def __len__(self):
        """
        The function returns the number of rows in the dataframe
        :return: The number of rows in the dataframe.
        """
        return self.df.shape[0]

    def _get_image(self, path) -> PIL.Image:
        """
        It takes a path to an image, opens it, converts it to RGB, and returns the image

        :param path: The path to the image you want to classify
        :return: The image data is being returned.
        """
        imageData = Image.open(path).convert("RGB")
        return imageData

    def __getitem__(self, index: int):
        """
        It takes an index as input, and returns a tuple of the image and the target label

        :param index: the index of the image in the dataset
        :type index: int
        :return: The image and the target
        """
        target = self.labels[index]
        img = self._get_image(self.df.loc[index]["image"])
        if self.transforms:
            img = self.transforms(img)
        return img, target


def prepare_dataloader(df: pd.DataFrame, trn_idx: np.ndarray, val_idx: np.ndarray):
    """
    > We take the training and validation indices, and use them to create two datasets, one for training
    and one for validation. We then create two dataloaders, one for training and one for validation

    :param df: the dataframe containing the image paths and labels
    :type df: pd.DataFrame
    :param trn_idx: the indices of the training set
    :type trn_idx: np.ndarray
    :param val_idx: the validation indices
    :type val_idx: np.ndarray
    :return: train_loader and val_loader
    """
    train_ = df.loc[trn_idx, :].reset_index(drop=True)
    valid_ = df.loc[val_idx, :].reset_index(drop=True)

    train_transforms, val_transforms, test_transforms = data_augmentations(img_size=224)

    train_ds = ChestXrayDataset(df=train_, transforms=train_transforms)
    valid_ds = ChestXrayDataset(df=valid_, transforms=train_transforms)

    from method_balance_data import weight_random_sampler

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=16,
        pin_memory=False,
        drop_last=False,
        shuffle=False,
        num_workers=4,
        # sampler=ImbalancedDatasetSampler(train_ds)
        sampler=weight_random_sampler(df=train_, train_ds=train_ds),
    )

    val_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=4,
        num_workers=4,
        shuffle=False,
        pin_memory=False,
        # sampler=ImbalancedDatasetSampler(valid_ds)
        sampler=weight_random_sampler(df=valid_, train_ds=valid_ds),
    )
    return train_loader, val_loader
