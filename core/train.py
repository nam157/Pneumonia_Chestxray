import torch
import time
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np

scaler = GradScaler()


def train_one_epoch(
    epoch,
    model,
    loss_fn,
    optimizer,
    train_loader,
    device,
    scheduler=None,
    schd_batch_update=False,
):
    """
    We're going to train the model for one epoch, and we're going to use the `scaler` to scale the loss,
    and then we're going to use the `scaler` to unscale the gradients, and then we're going to use the
    `scaler` to update the scale factor.

    Let's break it down.

    First, we're going to train the model for one epoch.

    :param epoch: the current epoch number
    :param model: the model to train
    :param loss_fn: the loss function to use
    :param optimizer: the optimizer to use
    :param train_loader: the training data loader
    :param device: the device to run on (CPU or GPU)
    :param scheduler: the scheduler object
    :param schd_batch_update: whether to update the scheduler after each batch or after each epoch,
    defaults to False (optional)
    """
    model.train()

    t = time.time()
    running_loss = None

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        with autocast():
            image_preds = model(imgs)
            loss = loss_fn(image_preds, image_labels)
            scaler.scale(loss).backward()

            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * 0.99 + loss.item() * 0.01

            if ((step + 1) % 10 == 0) or ((step + 1) == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if scheduler is not None and schd_batch_update:
                    scheduler.step()

            if ((step + 1) % 100 == 0) or ((step + 1) == len(train_loader)):
                description = f"epoch {epoch} loss: {running_loss:.4f}"
                pbar.set_description(description)

    if scheduler is not None and not schd_batch_update:
        scheduler.step()


def valid_one_epoch(
    epoch,
    model,
    loss_fn,
    val_loader,
    device,
    best_val,
    fold,
    scheduler=None,
    schd_loss_update=False,
):
    """
    It takes in the model, the loss function, the validation data loader, the device, the best
    validation accuracy, the fold number, the scheduler, and a boolean value that indicates whether the
    scheduler should be updated based on the loss or not.

    It then sets the model to evaluation mode, and then loops through the validation data loader.

    For each batch of images, it sends the images to the device, and then passes the images through the
    model to get the predictions.

    It then calculates the loss, and updates the loss sum and sample number.

    It then prints out the loss every 100 batches, and at the end of the epoch.

    It then concatenates all the predictions and targets, and calculates the validation accuracy.

    If the validation accuracy is greater than the best validation accuracy, it saves the model.

    :param epoch: the current epoch number
    :param model: the model we're training
    :param loss_fn: the loss function to use
    :param val_loader: the validation data loader
    :param device: the device to run the model on (CPU or GPU)
    :param best_val: the best validation accuracy so far
    :param fold: the fold number of the current training
    :param scheduler: the scheduler object
    :param schd_loss_update: if True, the scheduler will use the loss as the metric to update the
    learning rate, defaults to False (optional)
    :return: The best validation accuracy
    """
    model.eval()

    t = time.time()
    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        image_preds = model(imgs)  # output = model(input)
        # print(image_preds.shape, exam_pred.shape)
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]

        loss = loss_fn(image_preds, image_labels)

        loss_sum += loss.item() * image_labels.shape[0]
        sample_num += image_labels.shape[0]

        if ((step + 1) % 100 == 0) or ((step + 1) == len(val_loader)):
            description = f"epoch {epoch} loss: {loss_sum/sample_num:.4f}"
            pbar.set_description(description)

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    valid_acc = (image_preds_all == image_targets_all).mean()
    if valid_acc > best_val:
        best_val = valid_acc
        torch.save(model.state_dict(), f"./best_model_fold{fold}.pt")

    print("validation multi-class accuracy = {:.4f}".format(valid_acc))

    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum / sample_num)
        else:
            scheduler.step()

    return best_val
