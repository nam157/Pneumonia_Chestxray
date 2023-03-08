import pandas as pd
import torch
from data_loader import data_augmentations
from models import Classifier
from PIL import Image
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)


def infer(checkpoint, img_path, device):
    """
    It loads the model, loads the image, transforms the image, and then runs the model on the image

    :param checkpoint: the path to the saved model
    :param img_path: The path to the image you want to predict
    :param device: The device you want to run the model on
    :return: The model is returning the predicted class of the image.
    """
    model = Classifier(model_arch="densenet121", n_class=1, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    image = Image.open(img_path).convert("RGB")
    train_transforms, val_transforms, test_transforms = data_augmentations(img_size=224)
    img = test_transforms(image)
    img = img[None, :, :].to(device).float()
    output = model(img)
    print(output)
    pred = (output > 0.5).float()
    return pred.squeeze(1).detach().cpu().numpy()


def test(path_csv, checkpoint, device):
    """
    It takes in a path to a csv file, a checkpoint, and a device, and then it creates a new column in
    the csv file called 'preds' and populates it with the predictions from the model.

    :param path_csv: path to the csv file containing the image paths and labels
    :param checkpoint: the path to the saved model
    :param device: the device to run the model on
    """
    df = pd.read_csv(path_csv)
    df["preds"] = range(len(df))
    true = []
    preds = []
    for i in range(len(df)):
        img_path = df.loc[i, "image"]
        out = infer(checkpoint, img_path, device)
        print(out)
        df.loc[i, "preds"] = out[0]
        true.append(df.loc[i, "label"])
        preds.append(out[0])

    f1 = f1_score(y_true=true, y_pred=preds)
    precison = precision_score(y_true=true, y_pred=preds)
    recall = recall_score(y_true=true, y_pred=preds)
    confusionmatrix = confusion_matrix(y_true=true, y_pred=preds)
    df.to_csv("data_chestxray/preds.csv")

    print("----------------------------------------------------------")
    print(df["label"].value_counts())
    print("----------------------------------------------------------")
    print(df["preds"].value_counts())
    print(f"F1-SCORE: {f1} \t PRECISION: {precison} \t RECALL: {recall} ")
    print(f"Confunsion Matrix: {confusionmatrix}")


if __name__ == "__main__":
    device = torch.device("cpu")
    test(
        path_csv="data_chestxray/test.csv",
        checkpoint="saved_model/densenet121_fold_0_7.pt",
        device=device,
    )
    # pred = infer(checkpoint='saved_model/densenet121_fold_0_29.pt',img_path='chest_xray/test/NORMAL/IM-0111-0001.jpeg',device=device)
    # print(pred)
