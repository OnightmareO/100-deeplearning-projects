import torch
import torch.nn as nn

class SimplehzwClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super(SimplehzwClassifier, self).__init__()
        #对于 3x3 卷积，若使用 padding = (kernel_size - 1) // 2 = 1，则输出特征图的宽高会保持和输入一致。
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512 * 4 * 4, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes)
        )
    
    def forward(self, x):
        x = self.block1(x) #输入图像经过 block1 后，特征图的尺寸从 (3, 128, 128) 变为 (64, 64, 64)，通道数增加到 64，宽高减半。
        x = self.block2(x) #输入图像经过 block2 后，特征图的尺寸从 (64, 64, 64) 变为 (128, 32, 32)，通道数增加到 128，宽高减半。
        x = self.block3(x) #输入图像经过 block3 后，特征图的尺寸从 (128, 32, 32) 变为 (256, 16, 16)，通道数增加到 256，宽高减半。
        x = self.block4(x) #输入图像经过 block4 后，特征图的尺寸从 (256, 16, 16) 变为 (512, 8, 8)，通道数增加到 512，宽高减半。
        x = self.block5(x) #输入图像经过 block5 后，特征图的尺寸从 (512, 8, 8) 变为 (512, 4, 4)，通道数保持为 512，宽高减半。
        x = self.classifier(x)
        return x

            