import os

import numpy as np
import torch
from data_loader import concat_dataset


def compute_class_freqs(labels):
    N = labels.shape[0]

    positive_frequencies = np.sum(labels, 0) / N

    negative_frequencies = np.ones_like(positive_frequencies) - positive_frequencies

    return positive_frequencies, negative_frequencies


def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """
    Args:
      pos_weights (np.array): array of positive weights for each class, size (num_classes)
      neg_weights (np.array): array of negative weights for each class, size (num_classes)
    """

    def weighted_loss(y_true, y_pred):
        """
        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        """
        loss = 0.0
        for i in range(len(pos_weights)):
            print(y_pred[i])
            positive_term_loss = (
                pos_weights[i] * y_true[i] * torch.log(y_pred[i] + epsilon)
            )
            negative_term_loss = (
                neg_weights[i] * (1 - y_true[i]) * torch.log(1 - y_pred[i] + epsilon)
            )
            loss += -torch.mean(positive_term_loss + negative_term_loss)
        return loss

    return weighted_loss


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def weight_random_sampler(df, train_ds):
    from torch.utils.data import WeightedRandomSampler

    class_counts = df.label.value_counts()
    class_weights = 1 / class_counts
    sample_weights = [class_weights[i] for i in df.label.values]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_ds))

    return sampler


if __name__ == "__main__":
    data = concat_dataset(
        train_dir="data_chestxray/train",
        val_dir="data_chestxray/val",
        test_dir="data_chestxray/test.csv",
    )

    COUNT_PNEUMONIA, COUNT_NORMAL = data["label"].value_counts()
    print("Normal images count in training set: " + str(COUNT_NORMAL))
    print("Pneumonia images count in training set: " + str(COUNT_PNEUMONIA))
    print("Image count: " + str(len(data)))

    weight_for_0 = (1 / COUNT_NORMAL) * (len(data)) / 2.0
    weight_for_1 = (1 / COUNT_PNEUMONIA) * (len(data)) / 2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print(class_weight)

    labels = np.array(data["label"].values)
    freq_pos, freq_neg = compute_class_freqs(labels)
    pos_weights = freq_neg
    neg_weights = freq_pos

    print(pos_weights, neg_weights)

    pos_contribution = np.array(freq_pos) * pos_weights
    neg_contribution = np.array(freq_neg) * neg_weights

    print(pos_contribution, neg_contribution)
