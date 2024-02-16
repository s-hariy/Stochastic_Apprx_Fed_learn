import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import NUM_CLASSES


class LeNet5_FashionMNIST(nn.Module):
    def __init__(self):
        super(LeNet5_FashionMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet5_CIFAR10(nn.Module):
    def __init__(self):
        super(LeNet5_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) 
        self.pool = nn.MaxPool2d(2, 2)  
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)    

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes, dim=4608):
        super(ResNet9, self).__init__()
        self.prep = self._make_layer(in_channels, 64)
        self.layer1_head = self._make_layer(64, 128, pool=True)
        self.layer1_residual = nn.Sequential(
            self._make_layer(128, 128), 
            self._make_layer(128, 128)
        )
        self.layer2 = self._make_layer(128, 256, pool=True)
        self.layer3_head = self._make_layer(256, 512, pool=True)
        self.layer3_residual = nn.Sequential(
            self._make_layer(512, 512), 
            self._make_layer(512, 512)
        )
        self.MaxPool2d = nn.MaxPool2d(4)
        self.linear = nn.Linear(dim, num_classes)

    def _make_layer(self, in_channels, out_channels, pool=False):
        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1_head(x)
        x = self.layer1_residual(x) + x
        
        x = self.layer2(x)
        x = self.layer3_head(x)
        x = self.layer3_residual(x) + x
        
        x = self.MaxPool2d(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.linear(x)
        return x

class CIFAR100_net(nn.Module):
    def __init__(self):
        super(CIFAR100_net, self).__init__()
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Layer 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dropout = nn.Dropout(0.3)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.dropout3 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 100)  # 100 classes for CIFAR-100

    def forward(self, x):
        # Convolutional layers with ReLU, BatchNorm, MaxPool and Dropout
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        # x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))
        # x = self.dropout(x)

        # Flatten the output for the fully connected layers
        x = torch.flatten(x, 1)

        # Fully connected layer with ReLU and Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)

        # Output layer
        x = self.fc2(x)
        return x


class STL10_net(nn.Module):
    def __init__(self):
        super(STL10_net, self).__init__()
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 12 * 12, 1024)  # Adjusted for the reduced image size
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)  # 10 classes for STL-10

    def forward(self, x):
        # Apply first conv layer with pooling
        x = self.pool1(F.relu(self.conv1(x)))

        # Apply second conv layer with pooling
        x = self.pool2(F.relu(self.conv2(x)))

        # Apply third conv layer with pooling
        x = self.pool3(F.relu(self.conv3(x)))

        # Flatten the output for the fully connected layers
        x = x.view(-1, 256 * 12 * 12)  # Flatten

        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output layer
        x = self.fc3(x)
        return x


def get_model(dataset,arch):
    if dataset == "CIFAR10":
        if arch == "LeNet":
            model = LeNet5_CIFAR10()
        elif arch == "Resnet9":
            model = ResNet9(in_channels=3,num_classes=10,dim=512)
    elif dataset == "F-MNIST":
            model = LeNet5_FashionMNIST()

    elif dataset == "CIFAR100":
        if arch == "CNN":
            model = CIFAR100_net()
        elif arch == "Resnet9":
            model = ResNet9(in_channels=3,num_classes=100,dim=512)
    elif dataset == "STL10":
        if arch == "CNN":
            model = STL10_net()
        elif arch == "Resnet9":
            model = ResNet9(in_channels=3,num_classes=10,dim=4608)
        # elif arch = "Resnet18":
        #     model = resnet18()

    # for layer in model.children():
    #     if hasattr(layer, 'reset_parameters'):
    #         layer.reset_parameters()
    
    # model.apply(weight_init)
    return model
