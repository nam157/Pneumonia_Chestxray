from data_loader import (ChestXrayDataset, concat_dataset, create_data,
                         data_augmentations, prepare_dataloader)


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


def test_concat_dataset():
    dataset = concat_dataset(
        train_dir="chest_xray/train", val_dir="chest_xray/val", test_dir=None
    )
    print(dataset)


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
    import torch
    from torchsampler import ImbalancedDatasetSampler

    train_transforms, val_transforms, test_transforms = data_augmentations(img_size=224)

    train_ = create_data(data_dir="chest_xray/train", save_file="train.csv")
    train_ds = ChestXrayDataset(df=train_, transforms=train_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        sampler=ImbalancedDatasetSampler(train_ds),
        batch_size=1,
    )
    data, label = next(iter(train_loader))
    print(data.shape)


def test_data_undersampling():
    import torch
    from torch.utils.data import WeightedRandomSampler

    train_transforms, val_transforms, test_transforms = data_augmentations(img_size=224)
    train_ = create_data(data_dir="data_chestxray/train", save_file="train.csv")
    train_ds = ChestXrayDataset(df=train_, transforms=train_transforms)
    class_counts = train_.label.value_counts()
    print(class_counts)
    class_weights = 1 / class_counts
    print(class_weights)
    sample_weights = [class_weights[i] for i in train_.label.values]
    print(sample_weights[:10])
    print(train_.label.values[:10])
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_ds))

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        sampler=sampler,
        batch_size=1,
    )
    data, label = next(iter(train_loader))
    print(data.shape)


if __name__ == "__main__":
    test_data_undersampling()
