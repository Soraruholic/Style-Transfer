import torch
import torch.nn as nn
from torchvision import models, transforms

class vgg16(nn.Module):
    def __init__(self, weight_path = './weights/vgg16.pth'):
        super(vgg16, self).__init__()
        model = models.vgg16(pretrained = False)
        model.load_state_dict(torch.load(weight_path), strict = False)
        self.features = model.features

    def forward(self, x):
        feature_extract = {
            '3' : 'relu1_2',
            '8' : 'relu2_2',
            '15' : 'relu3_3',
            '22' : 'relu4_3'
        }
        out_features = {}
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in feature_extract:
                out_features[feature_extract[name]] = x
                if name == '22':
                    break
        return out_features


def vgg16_pretrained(weight_path = './weights/vgg16.pth'):
    model = vgg16(weight_path)

    # Fixed the pretrained loss network in order to define our loss functions
    for param in model.parameters():
        param.requires_grad = False
    return model