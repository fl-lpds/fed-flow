import torch.nn as nn

from app.model.interface.nn_model_interface import NNModel


# Build the VGG nn_model according to location and split_layer
class VGG16(NNModel):
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

        if self.location == 'Unit':  # Get the holistic nn_model
            pass

        for x in cfg:
            if x == 'layer1':
                self.layer1 = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU())
            if x == 'layer2':
                self.layer2 = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2))
            if x == 'layer3':
                self.layer3 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU())
            if x == 'layer4':
                self.layer4 = nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2))
            if x == 'layer5':
                self.layer5 = nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU())
            if x == 'layer6':
                self.layer6 = nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU())
            if x == 'layer7':
                self.layer7 = nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2))
            if x == 'layer8':
                self.layer8 = nn.Sequential(
                    nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU())
            if x == 'layer9':
                self.layer9 = nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU())
            if x == 'layer10':
                self.layer10 = nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2))
            if x == 'layer11':
                self.layer11 = nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU())
            if x == 'layer12':
                self.layer12 = nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU())
            if x == 'layer13':
                self.layer13 = nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2))
            if x == 'fc':
                self.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(7 * 7 * 512, 4096),
                    nn.ReLU())
            if x == 'fc1':
                self.fc1 = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(4096, 4096),
                    nn.ReLU())
            if x == 'fc2':
                self.fc2 = nn.Sequential(
                    nn.Linear(4096, num_classes))

        return None, None

    def _initialize_weights(self):
        pass

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def get_config(self):
        return ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'layer7', 'layer8', 'layer9', 'layer10',
                'layer11', 'layer12', 'layer13', 'fc', 'fc1', 'fc2']

    def get_representation(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)

        out = self.fc(out)
        out = self.fc1(out)
        return out
