import torch.nn as nn
from torchvision import transforms

from app.model.interface.nn_model_interface import NNModel


# Build the VGG nn_model according to location and split_layer
class vgg16(NNModel):
    def _make_layers(self, edge_based):
        num_classes = 10
        features = []
        denses = []
        cfg = self.get_config()
        if edge_based:
            if self.location == 'Server':
                cfg = cfg[self.split_layer[1] + 1:]

            if self.location == 'Client':
                cfg = cfg[:self.split_layer[0] + 1]

            if self.location == 'Edge':
                cfg = cfg[self.split_layer[0] + 1:self.split_layer[1] + 1]
        else:
            if self.location == 'Server':
                cfg = cfg[self.split_layer + 1:]

            if self.location == 'Client':
                cfg = cfg[:self.split_layer + 1]

        if self.location == 'Unit':
            pass

        for x in cfg:
            if x == 'layer1':
                self.layer1 = [
                    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU()]
                features += self.layer1
            if x == 'layer2':
                self.layer2 = [
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)]
                features += self.layer2
            if x == 'layer3':
                self.layer3 = [
                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU()]
                features += self.layer3
            if x == 'layer4':
                self.layer4 = [
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)]
                features += self.layer4
            if x == 'layer5':
                self.layer5 = [
                    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU()]
                features += self.layer5
            if x == 'layer6':
                self.layer6 = [
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU()]
                features += self.layer6
            if x == 'layer7':
                self.layer7 = [
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)]
                features += self.layer7
            if x == 'layer8':
                self.layer8 = [
                    nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU()]
                features += self.layer8
            if x == 'layer9':
                self.layer9 = [
                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU()]
                features += self.layer9
            if x == 'layer10':
                self.layer10 = [
                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)]
                features += self.layer10
            if x == 'layer11':
                self.layer11 = [
                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU()]
                features += self.layer11
            if x == 'layer12':
                self.layer12 = [
                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU()]
                features += self.layer12
            if x == 'layer13':
                self.layer13 = [
                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)]
                features += self.layer13
            if x == 'fc':
                self.fc = [
                    nn.Dropout(0.5),
                    nn.Linear(7 * 7 * 512, 4096),
                    nn.ReLU()]
                denses += self.fc
            if x == 'fc1':
                self.fc1 = [
                    nn.Dropout(0.5),
                    nn.Linear(4096, 4096),
                    nn.ReLU()]
                denses += self.fc1
            if x == 'fc2':
                self.fc2 = [
                    nn.Linear(4096, num_classes)]
                denses += self.fc2

        return nn.Sequential(*features), nn.Sequential(*denses)

    def _initialize_weights(self):
        pass

    def forward(self, x):
        if len(self.features) > 0:
            out = self.features(x)
        else:
            out = x
        if len(self.denses) > 0:
            out = out.view(out.size(0), -1)
            out = self.denses(out)

        return out

    def get_config(self):
        return ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'layer7', 'layer8', 'layer9', 'layer10',
                'layer11', 'layer12', 'layer13', 'fc', 'fc1', 'fc2']

    @staticmethod
    def data_transformer():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
