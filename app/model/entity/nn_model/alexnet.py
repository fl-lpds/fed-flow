import torch.nn as nn
import torchvision
from torchvision import transforms
from app.config.logger import fed_logger

from app.model.interface.nn_model_interface import NNModel


class alexnet(NNModel):

    def _make_layers(self, edge_based):
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

        if self.location == 'Unit':  # Get the holistic nn_model
            pass
        for x in cfg:
            if x == "layer1":
                self.layer1 = [
                    nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
                    nn.BatchNorm2d(96),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2)]
                features += self.layer1
            if x == "layer2":
                self.layer2 = [
                    nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2)]
                features += self.layer2
            if x == "layer3":
                self.layer3 = [
                    nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(384),
                    nn.ReLU()]
                features += self.layer3
            if x == "layer4":
                self.layer4 = [
                    nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(384),
                    nn.ReLU()]
                features += self.layer4
            if x == "layer5":
                self.layer5 = [
                    nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2)]
                features += self.layer5
            if x == "fc":
                self.fc = [
                    nn.Dropout(0.5),
                    nn.Linear(9216, 4096),
                    nn.ReLU()]
                denses += self.fc
            if x == "fc1":
                self.fc1 = [
                    nn.Dropout(0.5),
                    nn.Linear(4096, 4096),
                    nn.ReLU()]
                denses += self.fc1
            if x == "fc2":
                self.fc2 = [
                    nn.Linear(4096, 10)]
                denses += self.fc2
        return nn.Sequential(*features), nn.Sequential(*denses)

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
        if len(self.features) > 0:
            out = self.features(x)
        else:
            out = x
        if len(self.denses) > 0:
            out = out.view(out.size(0), -1)
            fed_logger.info(out.shape)
            out = self.denses(out)

        return out
    def get_config(self):
        return ["layer1", "layer2", "layer3", "layer4", "layer5", "fc", "fc1", "fc2"]

    @staticmethod
    def data_transformer():
        return transforms.transforms.Compose([
                transforms.Resize((227, 227)),
                transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            )
        ])
