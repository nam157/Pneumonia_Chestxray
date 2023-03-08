import os
import logging

import numpy as np
import torch
import wandb
from data_loader import concat_dataset, prepare_dataloader
from method_balance_data import FocalLoss, get_weighted_loss
from models import Classifier
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from train import train_one_epoch, valid_one_epoch

logging.basicConfig(
    filename="logs/train.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

def main():
    wdb = wandb.init(
        project="pneumonia_chestxray",
        config={
            "optimizer": "AdamW",
            "learning_rate": 0.0005,
            "architecture": "mobilenetv3_large_100",
            "epochs": 30,
            "Loss": "BCEWithLogitsLoss",
        },
    )
    logging.info("Create dataset")
    device = torch.device("cuda")
    dataset = concat_dataset(
        train_dir="data_chestxray/train", val_dir="data_chestxray/val", test_dir=None
    )
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=1).split(
        np.arange(dataset.shape[0]), dataset.label.values
    )
    for fold, (trn_idx, val_idx) in enumerate(folds):
        if fold > 0:
            break

        train_loader, val_loader = prepare_dataloader(
            df=dataset, trn_idx=trn_idx, val_idx=val_idx
        )
        logging.info("Load model mobilenetv3_large_100")
        model = Classifier(
            model_arch="mobilenetv3_large_100", n_class=1, pretrained=True
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

        # loss = get_weighted_loss(pos_weights=[0.2703210382513661],neg_weights=[0.7296789617486339])
        # loss = FocalLoss(gamma=0.5,alpha=0.25).to(device)
        loss = nn.BCELoss().to(device)

        for epoch in range(30):
            running_loss = train_one_epoch(
                epoch=epoch,
                model=model,
                loss_fn=loss,
                optimizer=optimizer,
                train_loader=train_loader,
                device=device,
            )
            description = f"epoch {epoch} loss: {running_loss:.4f}"
            logging.info("TRAIN: {}".format(description))
            validation_loss = valid_one_epoch(
                epoch=epoch,
                model=model,
                loss_fn=loss,
                val_loader=val_loader,
                device=device,
                logger=logging,
            )
            description = f"epoch {epoch} loss: {validation_loss:.4f}"
            wdb.log({"loss-train": running_loss, "loss-valid": validation_loss})
            logging.info("VALID: {}".format(description))
            scheduler.step()

            os.makedirs("./saved_model",exist_ok=True)
            torch.save(
                model.state_dict(),
                "saved_model/{}_fold_{}_{}.pt".format(
                    "mobilenetv3_large_100", fold, epoch
                ),
            )

        del model, optimizer, train_loader, val_loader, scheduler
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
