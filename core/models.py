import timm
import torch.nn as nn


# > We're creating a classifier that takes in a model architecture, number of classes, and a
# pretrained flag.
#
# The classifier class inherits from the nn.Module class. This is a class that PyTorch provides to
# help us create our own neural network modules.
#
# The classifier class has two methods:
#
# 1. __init__: This is the constructor method. It's called when we create an instance of the
# classifier class.
# 2. forward: This is the method that's called when we pass data through the classifier.
#
# Let's look at the __init__ method first.
#
# The first line of the __init__ method is:
class Classifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x
