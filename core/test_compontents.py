from data_loader import create_data,ChestXrayDataset,data_augmentations,prepare_dataloader

def test_create_dataset():
    data = create_data(data_dir="data_chestxray/train/", save_file="train.csv")
    train_transforms,val_transforms,test_transforms = data_augmentations(img_size=224)
    dataset = ChestXrayDataset(df=data, transforms=train_transforms)
    img, target = dataset[0]
    print('label:',target)
    print("shape: ",img.shape)


def test_data_loader():
    data_train = create_data(data_dir="data_chestxray/train/", save_file="train.csv")
    data_val = create_data(data_dir="data_chestxray/val/", save_file="val.csv")
    train_loader,val_loader = prepare_dataloader(df_train= data_train,df_val=data_val)
    print(next(iter(train_loader)))