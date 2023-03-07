import pandas as pd
from PIL import Image
import torch
from data_loader import data_augmentations
from models import Classifier

def infer(checkpoint,img_path,device):
    """
    It loads the model, loads the image, transforms the image, and then runs the model on the image
    
    :param checkpoint: the path to the saved model
    :param img_path: The path to the image you want to predict
    :param device: The device you want to run the model on
    :return: The model is returning the predicted class of the image.
    """
    model = Classifier(model_arch="densenet121", n_class=2, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint,map_location=device))
    image = Image.open(img_path).convert("RGB")
    train_transforms, _, test_transforms = data_augmentations(img_size=224)
    img = train_transforms(image)
    img = img[None,:,:].to(device).float()
    output = model(img)
    pred = torch.argmax(output, 1).detach().cpu().numpy()
    return pred

def test(path_csv,checkpoint,device):
    """
    It takes in a path to a csv file, a checkpoint, and a device, and then it creates a new column in
    the csv file called 'preds' and populates it with the predictions from the model.
    
    :param path_csv: path to the csv file containing the image paths and labels
    :param checkpoint: the path to the saved model
    :param device: the device to run the model on
    """
    df = pd.read_csv(path_csv)
    df['preds'] = range(len(df))
    for i in range(len(df)):
        img_path = df.loc[i,'image']
        out = infer(checkpoint,img_path,device)
        df['preds'] = out[0]
    df.to_csv('data_chestxray/preds.csv')

if __name__ == "__main__":
    device = torch.device('cuda')
    test(path_csv='data_chestxray/test.csv',checkpoint='densenet121_fold_0_3',device=device)

    
