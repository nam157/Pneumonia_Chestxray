import os 
import numpy as np
from data_loader import concat_dataset
import torch

data = concat_dataset()
COUNT_PNEUMONIA,COUNT_NORMAL = data['label'].value_counts()
print("Normal images count in training set: " + str(COUNT_NORMAL))
print("Pneumonia images count in training set: " + str(COUNT_PNEUMONIA))
print("Image count: " + str(len(data)))

weight_for_0 = (1 / COUNT_NORMAL)*(len(data))/2.0 
weight_for_1 = (1 / COUNT_PNEUMONIA)*(len(data))/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print(class_weight)

def compute_class_freqs(labels):
    """
    Compute positive and negative frequences for each class.

    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes)
    Returns:
        positive_frequencies (np.array): array of positive frequences for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequences for each
                                         class, size (num_classes)
    """

    N = labels.shape[0]
    
    print('Pneumonia:',np.sum(labels,0))
    print("Normal:",N-np.sum(labels,0))

    positive_frequencies = np.sum(labels,0)/N
    
    negative_frequencies = np.ones_like(positive_frequencies) - positive_frequencies

    return positive_frequencies, negative_frequencies

labels = np.array(data['label'].values)
freq_pos, freq_neg = compute_class_freqs(labels)
pos_weights = freq_neg
neg_weights = freq_pos

print(pos_weights,neg_weights)

pos_contribution = np.array(freq_pos) * pos_weights 
neg_contribution = np.array(freq_neg) * neg_weights

print(pos_contribution,neg_contribution)

def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """
    Return weighted loss function given negative weights and positive weights.

    Args:
      pos_weights (np.array): array of positive weights for each class, size (num_classes)
      neg_weights (np.array): array of negative weights for each class, size (num_classes)
    
    Returns:
      weighted_loss (function): weighted loss function
    """
    def weighted_loss(y_true, y_pred):
        """
        Return weighted loss value. 

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (Tensor): overall scalar loss summed across all classes
        """
        loss = 0.0
        for i in range(len(pos_weights)):
            print(y_pred[i])
            positive_term_loss = pos_weights[i]*y_true[i]*torch.log(y_pred[i] + epsilon)
            negative_term_loss = neg_weights[i]*(1-y_true[i])*torch.log(1-y_pred[i] + epsilon)
            loss += -torch.mean(positive_term_loss + negative_term_loss)  
        return loss
    
    return weighted_loss