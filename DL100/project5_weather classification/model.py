import torch
import torch.nn as nn
import torch.optim as optim

class SimpleWeatherClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleWeatherClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 56 * 56, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, num_classes)
        

    def forward(self, x):
        x = self.relu(self.conv1(x)) # 224x224x3 -> 224x224x16
        x = self.pool(x) # 224x224x16 -> 112x112x16
        x = self.relu(self.conv2(x)) # 112x112x16 -> 112x112x32
        x = self.pool(x) # 112x112x32 -> 56x56x32
        x = self.relu(self.conv3(x)) # 56x56x32 -> 56x56x64
        x = self.dropout(x) # 56x56x64 -> 56x56x64
        x = self.flatten(x) 
        x = self.relu(self.fc1(x)) # 56*56*64 -> 2048
        x = self.relu(self.fc2(x)) # 2048 -> 512
        x = self.fc3(x) # 512 -> num_classes          
        return x