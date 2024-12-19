import torch.nn as nn
import torchvision
from torchvision import transforms

from app.model.interface.nn_model_interface import NNModel


class resnet50(NNModel):
    def __init__(self):
        super(resnet50, self).__init__()
        self.model = torchvision.models.resnet50()
        self.cfg = self.get_config()



    def _make_layers(self, edge_based):
        layers = list(self.model.children())
        sub_layers = layers
        if edge_based:
            if self.location == 'Server':
                sub_layers = layers[self.split_layer[1] + 1:]

            if self.location == 'Client':
                sub_layers = layers[:self.split_layer[0] + 1]

            if self.location == 'Edge':
                sub_layers = layers[self.split_layer[0] + 1:self.split_layer[1] + 1]
        else:
            if self.location == 'Server':
                sub_layers = layers[self.split_layer + 1:]

            if self.location == 'Client':
                sub_layers = layers[:self.split_layer + 1]

        if self.location == 'Unit':  # Get the holistic nn_model
            pass
        submodel = nn.Sequential(*sub_layers)

        return submodel, None

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.features(x)
        return out

    def get_config(self):
        return list(self.model.children())

    @staticmethod
    def data_transformer():
        return transforms.Compose([
            transforms.Resize(256),  # Resize the shorter side to 256
            transforms.CenterCrop(224),  # Center crop to 224x224
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(  # Normalize using ImageNet stats
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
