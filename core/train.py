import time

import numpy as np
import torch
from models import Classifier
from torch.cuda.amp import autocast
from tqdm import tqdm
from sklearn.metrics import f1_score,recall_score,precision_score

def train_one_epoch(
    epoch: int,
    model: Classifier,
    loss_fn,
    optimizer: torch.optim,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """
    > We iterate through the training data, and for each batch, we pass the images through the model,
    calculate the loss, and update the model parameters

    :param epoch: the current epoch number
    :param model: the model we're training
    :param loss_fn: the loss function to use
    :param optimizer: The optimizer to use for training
    :param train_loader: the training data loader
    :param device: the device to run the training on
    :return: The average loss per batch
    """

    model.train()

    running_loss = []

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        optimizer.zero_grad()
        image_preds = model(imgs)
        loss = loss_fn(image_preds, image_labels) 
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())
    return sum(running_loss) / len(running_loss)
            


def valid_one_epoch(
    epoch: int,
    model: Classifier,
    loss_fn,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    best_val: float,
    fold: int,
):
    """
    It takes in the model, the loss function, the validation data loader, the device, the best
    validation accuracy, and the fold number.

    It then runs the model on the validation data, and calculates the validation loss and accuracy.

    If the validation accuracy is better than the best validation accuracy, it saves the model.

    It returns the best validation accuracy.

    :param epoch: the current epoch number
    :param model: the model we're training
    :param loss_fn: the loss function we're using
    :param val_loader: the validation data loader
    :param device: the device to run the training on
    :param best_val: the best validation accuracy we've seen so far
    :param fold: the fold number
    :return: The best validation accuracy
    """
    model.eval()
    validation_loss = 0.0
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        image_preds = model(imgs)
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]

        loss = loss_fn(image_preds, image_labels)

        validation_loss += loss.item()

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)

    print(f"image_preds_all: {image_preds_all}")
    print(f"image_targets_all: {image_targets_all}")

    valid_acc = (image_preds_all == image_targets_all).mean()
    valid_f1  = f1_score(image_targets_all,image_preds_all)
    valid_precison = precision_score(image_targets_all,image_preds_all)
    valid_recall  = recall_score(image_targets_all,image_preds_all)


    if valid_acc > best_val:
        best_val = valid_acc
        torch.save(model.state_dict(), f"./best_model_fold{fold}.pt")

    print("validation  accuracy = {:.4f}".format(valid_acc))
    print("Validation F1-Score = {:.4f}".format(valid_f1))
    print("Validation Pecision-Score = {:.4f}".format(valid_precison))
    print("Validation Recall-Score = {:.4f}".format(valid_recall))

    return validation_loss / len(val_loader),best_val
