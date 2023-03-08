import numpy as np
import matplotlib.pyplot as plt


def visualise_dataloader(dl, id_to_label=None, with_outputs=True):
    total_num_images = len(dl.dataset)
    idxs_seen = []
    class_0_batch_counts = []
    class_1_batch_counts = []

    for i, batch in enumerate(dl):

        idxs = batch[0][:, 0].tolist()
        classes = batch[0][:, 1]
        class_ids, class_counts = classes.unique(return_counts=True)
        class_ids = set(class_ids.tolist())
        class_counts = class_counts.tolist()

        idxs_seen.extend(idxs)

        if len(class_ids) == 2:
            class_0_batch_counts.append(class_counts[0])
            class_1_batch_counts.append(class_counts[1])
        elif len(class_ids) == 1 and 0 in class_ids:
            class_0_batch_counts.append(class_counts[0])
            class_1_batch_counts.append(0)
        elif len(class_ids) == 1 and 1 in class_ids:
            class_0_batch_counts.append(0)
            class_1_batch_counts.append(class_counts[0])
        else:
            raise ValueError("More than two classes detected")

    if with_outputs:
        fig, ax = plt.subplots(1, figsize=(15, 15))

        ind = np.arange(len(class_0_batch_counts))
        width = 0.35

        ax.bar(
            ind,
            class_0_batch_counts,
            width,
            label=(id_to_label[0] if id_to_label is not None else "0"),
        )
        ax.bar(
            ind + width,
            class_1_batch_counts,
            width,
            label=(id_to_label[1] if id_to_label is not None else "1"),
        )
        ax.set_xticks(ind, ind + 1)
        ax.set_xlabel("Batch index", fontsize=12)
        ax.set_ylabel("No. of images in batch", fontsize=12)
        ax.set_aspect("equal")

        plt.legend()
        plt.show()

        num_images_seen = len(idxs_seen)

        print(
            f'Avg Proportion of {(id_to_label[0] if id_to_label is not None else "Class 0")} per batch: {(np.array(class_0_batch_counts) / 10).mean()}'
        )
        print(
            f'Avg Proportion of {(id_to_label[1] if id_to_label is not None else "Class 1")} per batch: {(np.array(class_1_batch_counts) / 10).mean()}'
        )
        print("=============")
        print(f"Num. unique images seen: {len(set(idxs_seen))}/{total_num_images}")
    return class_0_batch_counts, class_1_batch_counts, idxs_seen

if __name__ == "__main__":
    from data_loader import ChestXrayDataset, concat_dataset, create_data,prepare_dataloader
    import torch
    from torch.utils.data import TensorDataset,DataLoader

    data_train = create_data(data_dir="data_chestxray/test/", save_file="train.csv")

    label_to_id = {v: idx for idx, v in enumerate(data_train.label.unique())}
    num_classes = len(label_to_id)
    ds = TensorDataset(torch.as_tensor([(idx,label_to_id[l]) for idx,l in enumerate(data_train.label.values)]))
    dl = DataLoader(ds,shuffle=True,batch_size=10)
    visualise_dataloader(dl,label_to_id)

    from torch.utils.data import WeightedRandomSampler

    class_counts = data_train.label.value_counts()
    class_weights = 1 / class_counts
    sample_weights = [class_weights[i] for i in data_train.label.values]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(data_train))
    dl = DataLoader(ds,batch_size=10,sampler=sampler)
    visualise_dataloader(dl,label_to_id)
