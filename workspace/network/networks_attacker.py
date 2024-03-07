import torch
import torch.nn as nn
import torch.nn.functional as F

class AttackerNetwork_MNIST(nn.Module):
    def __init__(self):
        super(AttackerNetwork_MNIST, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 400)
        self.fc2 = nn.Linear(400, 10)
        self.dropout_input = 0.0
        self.dropout_hidden = 0.0
        self.is_training = True
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.dropout(x, p=self.dropout_input, training=self.is_training)
        x = F.dropout(F.relu(self.fc1(x)), p=self.dropout_hidden, training=self.is_training)
        x = self.fc2(x)
        return x

# 頂いたコードをベースに構築
class AttackerNetworks_MNIST_M(nn.Module):
    def __init__(self, classes):
        super(AttackerNetworks_MNIST_M, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16,out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(16*7*7, classes),  # Assuming input images are 28x28
        )

    def forward(self, x):
        return self.layers(x)
    
class AttackerNetworks_MNIST_S(nn.Module):
    def __init__(self, classes):
        super(AttackerNetworks_MNIST_S, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(16*14*14, classes),  # Assuming input images are 28x28
        )

    def forward(self, x):
        return self.layers(x)

    
class AttackerNetworks_MNIST_L(nn.Module):
    def __init__(self, classes):
        super(AttackerNetworks_MNIST_L, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(16*3*3, 512),  # Assuming input images are 28x28
            nn.ReLU(),
            nn.Linear(512, classes)  # Assuming 10 classes for MNIST
        )

    def forward(self, x):
        return self.layers(x)

    
class AttackerNetwork_MedMNIST(nn.Module):
    def __init__(self):
        super(AttackerNetwork_MedMNIST, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            # nn.Flatten(),
            # nn.Linear(16*3*3, 512),  # Assuming input images are 28x28
            nn.Flatten(),
            nn.Linear(32*7*7, 512),  # For usual style
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(256, 10)  # Assuming 10 classes for MNIST
        )

    def forward(self, x):
        return self.layers(x)
    
class AttackerNetworkSmall_MedMNIST(nn.Module):
    def __init__(self):
        super(AttackerNetworkSmall_MedMNIST, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16,out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(16*7*7, 10),  # Assuming input images are 28x28
        )

    def forward(self, x):
        return self.layers(x)