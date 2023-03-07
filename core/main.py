from tqdm import tqdm
import numpy as np
import torch
from data_loader import concat_dataset, prepare_dataloader
from models import Classifier
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.cuda.amp import GradScaler
from train import train_one_epoch, valid_one_epoch
from method_balance_data import get_weighted_loss

def main():
    device = torch.device("cuda")
    dataset = concat_dataset(train_dir="chest_xray/train", val_dir="chest_xray/val")
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=121).split(
        np.arange(dataset.shape[0]), dataset.label.values
    )
    for fold, (trn_idx, val_idx) in enumerate(folds):
        if fold > 0:
            break

        train_loader, val_loader = prepare_dataloader(
            df=dataset, trn_idx=trn_idx, val_idx=val_idx
        )
        model = Classifier(model_arch="densenet121", n_class=2, pretrained=True).to(
            device
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=10 - 1)

        loss = get_weighted_loss(pos_weights=[0.2703210382513661],neg_weights=[0.7296789617486339])

        # loss = nn.CrossEntropyLoss().to(device)
        
        
        best_val = 0

        for epoch in range(10):
            running_loss = train_one_epoch(epoch=epoch,model=model,loss_fn=loss,optimizer=optimizer,train_loader=train_loader,device=device)
            description = f"epoch {epoch} loss: {running_loss:.4f}"
            print(f'TRAIN: {description}')
            validation_loss,acc_val = valid_one_epoch(epoch=epoch,model=model,loss_fn=loss,val_loader=val_loader,device=device,best_val=best_val,fold=fold)
            description = f"epoch {epoch} loss: {validation_loss:.4f}"
            print(f'VALID: {description}')
            scheduler.step()
            torch.save(model.state_dict(), "{}_fold_{}_{}.pt".format("densenet121", fold, epoch))

        del model, optimizer, train_loader, val_loader, scheduler
        torch.cuda.empty_cache()



if __name__ == "__main__":
    main()