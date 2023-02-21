# initialize model

import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict

class Model:
    model = torchvision.models.vgg16(pretrained=True)
    n_inputs = model.classifier[6].in_features
    classification_layer = nn.Linear(n_inputs, 10)
    model.classifier[6] = classification_layer

    def __init__(self):
        self.load_model()

    def load_model(self):
        self.model.load_state_dict(torch.load("models\model.pth", map_location=torch.device('cpu'))) #torch
        self.model.eval()

    @property
    def get_model(self):
        return self.model