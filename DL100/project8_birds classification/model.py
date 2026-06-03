import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4 #扩展系数，表示输出通道数是输入通道数的多少倍

    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample #下采样模块，用于调整残差连接的输入特征图的尺寸，使其与卷积层的输出特征图匹配
        self.stride = stride

    def forward(self, x):
        identity = x #保存输入的特征图作为残差连接的输入

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x) #如果需要下采样，则对输入的特征图进行下采样，使其与输出的特征图尺寸匹配

        out += identity #将残差连接的输入与卷积层的输出相加
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,block,layers,num_classes):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.block = block #Bottleneck 模块类，用于构建残差块
        self.layers = layers #一个列表，表示每个残差块中 Bottleneck 模块的数量。

        # ResNet 的第一层是一个卷积层，输入通道数为 3，输出通道数为 64，卷积核大小为 7x7，步幅为 2，填充为 3，且不使用偏置项
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet 的主体部分由四个残差块组成，每个残差块包含多个 Bottleneck 模块。
        # 每个残差块的输出通道数分别为 64、128、256 和 512，且每个残差块的第一个 Bottleneck 模块的步幅为 2，以实现下采样。
        self.layer1 = self._make_layer(self.block, 64, block_num=self.layers[0], stride=1)
        self.layer2 = self._make_layer(self.block, 128, block_num=self.layers[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, block_num=self.layers[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, block_num=self.layers[3], stride=2)

        # ResNet 的最后部分是一个全局平均池化层和一个全连接层。
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, planes, block_num, stride=1):

        downsample=None
        #如果步幅不为 1 或输入通道数不等于输出通道数乘以扩展系数，则需要进行下采样，以确保残差连接的输入和输出具有相同的尺寸。
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        #第一个 Bottleneck 模块可能需要进行下采样，以调整特征图的尺寸。
        conv_block = block(self.inplanes, planes, stride, downsample)
        layers.append(conv_block)
        #更新输入通道数为输出通道数乘以扩展系数，以便后续的 Bottleneck 模块使用正确的输入通道数。
        self.inplanes = planes * block.expansion
        #后续的 Bottleneck 模块不需要进行下采样，因为它们的输入和输出尺寸已经匹配。
        for _ in range(1, block_num):
            layers.append(block(self.inplanes, planes, stride=1))
   
        #将所有的 Bottleneck 模块组合成一个 Sequential 模块，并返回该模块作为残差块的输出。
        return nn.Sequential(*layers)


if __name__ == "__main__":
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1000)
    print(model)

    