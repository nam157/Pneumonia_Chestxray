from data_loader import (
    ChestXrayDataset,
    create_data,
    data_augmentations,
    prepare_dataloader,
)


def test_create_dataset():
    """
    It takes in a dataframe and a list of transforms and returns a dataset object
    """
    data = create_data(data_dir="chest_xray/test/", save_file="test.csv")
    train_transforms, val_transforms, test_transforms = data_augmentations(img_size=224)
    dataset = ChestXrayDataset(df=data, transforms=train_transforms)
    img, target = dataset[0]
    print("label:", target)
    print("shape: ", img.shape)


def test_data_loader():
    """
    It takes in the data directory and the name of the file to save the data to, and returns a dataframe
    with the image paths and labels
    """
    data_train = create_data(data_dir="data_chestxray/train/", save_file="train.csv")
    data_val = create_data(data_dir="data_chestxray/val/", save_file="val.csv")
    train_loader, val_loader = prepare_dataloader(df_train=data_train, df_val=data_val)
    print(next(iter(train_loader)))

def test_data_loader_oversampling():
    from torchsampler import ImbalancedDatasetSampler
    import torch
    train_transforms, val_transforms, test_transforms = data_augmentations(img_size=224)

    train_ = create_data(data_dir="data_chestxray/test/", save_file="test.csv")
    train_ds = ChestXrayDataset(df=train_, transforms=train_transforms)
    print(ImbalancedDatasetSampler(train_ds))
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        sampler=ImbalancedDatasetSampler(train_ds),
        batch_size=16,
    )
    print(next(iter(train_loader)))

if __name__ == "__main__":
    test_data_loader_oversampling()