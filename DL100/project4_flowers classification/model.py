import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SimpleflowersClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleflowersClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1) 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 14 * 14, 512)  # Assuming input images are resized to 128x128
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x)) #(3,128,128) -> (16,126,126)
        x = self.pool(x) # (16,126,126) -> (16,63,63)
        x = self.relu(self.conv2(x)) # (16,63,63) -> (32,61,61)
        x = self.pool(x) # (32,61,61) -> (32,30,30)
        x = self.relu(self.conv3(x)) # (32,30,30) -> (64,28,28)
        x = self.pool(x) # (64,28,28) -> (64,14,14)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x