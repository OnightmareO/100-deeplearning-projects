import torch
import torch.nn as nn


class SimpleFashionMnist(nn.Module):
    def __init__(self):
        super(SimpleFashionMnist, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        #输入的图像是28x28x1的灰度图像
        x = self.relu(self.conv1(x)) #torch.Size([64, 16, 26, 26])
        x = self.relu(self.conv2(x)) #torch.Size([64, 64, 11, 11])
        x = self.pool(x) #torch.Size([64, 64, 5, 5])
        x = self.relu(self.conv3(x)) #torch.Size([64, 128, 3, 3])
        x = self.pool(x) #torch.Size([64, 128, 1, 1])
        x = self.flatten(x) 
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    


