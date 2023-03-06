import torch
from sklearn.model_selection import StratifiedKFold
from train import train_one_epoch, valid_one_epoch
from models import Classifier
from torch.cuda.amp import GradScaler
from torch import nn
import numpy as np
from data_loader import prepare_dataloader, create_data


if __name__ == "__main__":
    train = create_data(path_csv="data_chestxray/train.csv")
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=121).split(
        np.arange(train.shape[0]), train.target.values
    )

    for fold, (trn_idx, val_idx) in enumerate(folds):
        if fold > 0:
            break

        print("Training with {} started".format(fold))

        print(len(trn_idx), len(val_idx))
        train_loader, val_loader = prepare_dataloader(
            df_train=data_train, df_val=data_val
        )

        device = torch.device("cuda")

        model = Classifier(
            model_arch="densenet121", n_class=train.target.nunique(), pretrained=True
        ).to(device)
        scaler = GradScaler()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, gamma=0.1, step_size=10 - 1
        )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG['T_0'], T_mult=1, eta_min=CFG['min_lr'], last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=25,
        #                                                max_lr=CFG['lr'], epochs=CFG['epochs'], steps_per_epoch=len(train_loader))

        loss_tr = nn.CrossEntropyLoss().to(device)  # MyCrossEntropyLoss().to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)

        best_val = 0

        for epoch in range(10):
            train_one_epoch(
                epoch,
                model,
                loss_tr,
                optimizer,
                train_loader,
                device,
                scheduler=scheduler,
                schd_batch_update=False,
            )

            with torch.no_grad():
                best_val = valid_one_epoch(
                    epoch,
                    model,
                    loss_fn,
                    val_loader,
                    device,
                    best_val,
                    fold,
                    scheduler=None,
                    schd_loss_update=False,
                )

            torch.save(
                model.state_dict(), "{}_fold_{}_{}".format("densenet121", fold, epoch)
            )

        # torch.save(model.cnn_model.state_dict(),'{}/cnn_model_fold_{}_{}'.format(CFG['model_path'], fold, CFG['tag']))
        del model, optimizer, train_loader, val_loader, scaler, scheduler
        torch.cuda.empty_cache()
