import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------- 基础卷积单元 Conv+BN+ReLU --------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# -------------------------- Block35 (A模块，浅层残差块) --------------------------
class Block35(nn.Module):
    def __init__(self, channels, scale=0.17):
        super().__init__()
        self.scale = scale
        # 分支1: 1x1
        self.branch1 = ConvBNReLU(channels, 32, kernel_size=1)
        # 分支2: 1x1 -> 3x3
        self.branch2 = nn.Sequential(
            ConvBNReLU(channels, 32, kernel_size=1),
            ConvBNReLU(32, 32, kernel_size=3, padding=1)
        )
        # 分支3: 1x1 -> 3x3 -> 3x3
        self.branch3 = nn.Sequential(
            ConvBNReLU(channels, 32, kernel_size=1),
            ConvBNReLU(32, 48, kernel_size=3, padding=1),
            ConvBNReLU(48, 48, kernel_size=3, padding=1)
        )
        # 拼接后1x1融合通道
        self.merge_conv = nn.Conv2d(32+32+48, channels, kernel_size=1, bias=False)

    def forward(self, x):
        residual = torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x)
        ], dim=1)
        residual = self.merge_conv(residual)
        # 残差缩放相加
        out = x + self.scale * residual
        return F.relu(out)

# -------------------------- Reduction A 第一次下采样降维 --------------------------
class ReductionA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 分支1: 3x3 stride=2
        self.branch1 = ConvBNReLU(in_channels, 384, kernel_size=3, stride=2)
        # 分支2: 1x1 -> 3x3 -> 3x3 stride=2
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, 192, kernel_size=1),
            ConvBNReLU(192, 224, kernel_size=3, padding=1),
            ConvBNReLU(224, 256, kernel_size=3, stride=2)
        )
        # 分支3: MaxPool下采样
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b1, b2, b3], dim=1)

# -------------------------- Block17 (B模块，中层7x7分解残差块) --------------------------
class Block17(nn.Module):
    def __init__(self, channels, scale=0.17):
        super().__init__()
        self.scale = scale
        # 分支1:1x1
        self.branch1 = ConvBNReLU(channels, 192, kernel_size=1)
        # 分支2:1x1 -> (1,7)+(7,1) 分解卷积
        self.branch2 = nn.Sequential(
            ConvBNReLU(channels, 128, kernel_size=1),
            ConvBNReLU(128, 160, kernel_size=(1,7), padding=(0,3)),
            ConvBNReLU(160, 192, kernel_size=(7,1), padding=(3,0))
        )
        self.merge_conv = nn.Conv2d(192+192, channels, kernel_size=1, bias=False)

    def forward(self, x):
        residual = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        residual = self.merge_conv(residual)
        out = x + self.scale * residual
        return F.relu(out)

# -------------------------- Reduction B 第二次下采样降维 --------------------------
class ReductionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 分支1:1x1->3x3 stride2
        self.branch1 = nn.Sequential(
            ConvBNReLU(in_channels, 192, kernel_size=1),
            ConvBNReLU(192, 320, kernel_size=3, stride=2)
        )
        # 分支2:1x1->1x7+7x1->3x3 stride2
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, 192, kernel_size=1),
            ConvBNReLU(192, 192, kernel_size=(1,7), padding=(0,3)),
            ConvBNReLU(192, 192, kernel_size=(7,1), padding=(3,0)),
            ConvBNReLU(192, 192, kernel_size=3, stride=2)
        )
        # 分支3 maxpool
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b1, b2, b3], dim=1)

# -------------------------- Block8 (C模块，最后层3x3拆分残差块) --------------------------
class Block8(nn.Module):
    def __init__(self, channels, scale=0.17, no_relu=False):
        super().__init__()
        self.scale = scale
        self.no_relu = no_relu
        # 分支1:1x1
        self.branch1 = ConvBNReLU(channels, 192, kernel_size=1)
        # 分支2:1x1 -> (1,3)+(3,1)
        self.branch2 = nn.Sequential(
            ConvBNReLU(channels, 192, kernel_size=1),
            ConvBNReLU(192, 224, kernel_size=(1,3), padding=(0,1)),
            ConvBNReLU(224, 256, kernel_size=(3,1), padding=(1,0))
        )
        self.merge_conv = nn.Conv2d(192+256, channels, kernel_size=1, bias=False)

    def forward(self, x):
        residual = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        residual = self.merge_conv(residual)
        out = x + self.scale * residual
        if not self.no_relu:
            out = F.relu(out)
        return out

# -------------------------- 辅助分类头 AuxLogits --------------------------
class AuxLogits(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = ConvBNReLU(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(128 * 5 * 5, 1024)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# -------------------------- 完整 Inception-ResNet-v2 主网络 --------------------------
class InceptionResNetV2(nn.Module):
    def __init__(self, num_classes=58, aux_logits=True, dropout_rate=0.2):
        super().__init__()
        self.aux_logits = aux_logits

        # 输入头部 Stem
        self.stem = nn.Sequential(
            ConvBNReLU(3, 32, kernel_size=3, stride=2),
            ConvBNReLU(32, 32, kernel_size=3),
            ConvBNReLU(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ConvBNReLU(64, 80, kernel_size=1),
            ConvBNReLU(80, 192, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # 10个 Block35(A)
        self.blocks35 = nn.Sequential(*[Block35(192) for _ in range(10)])

        # 第一次降维
        self.reduction_a = ReductionA(192)

        # 20个 Block17(B)
        self.blocks17 = nn.Sequential(*[Block17(832) for _ in range(20)])

        # 辅助分支挂载在Block17末尾
        if self.aux_logits:
            self.aux = AuxLogits(832, num_classes)

        # 第二次降维
        self.reduction_b = ReductionB(832)

        # 9个 Block8(C) + 1个无ReLU的Block8收尾
        self.blocks8 = nn.Sequential(
            *[Block8(1344) for _ in range(9)],
            Block8(1344, no_relu=True)
        )

        # 全局池化与分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(1344, num_classes)

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Block35 A组
        x = self.blocks35(x)

        # Reduction A
        x = self.reduction_a(x)

        # Block17 B组
        x = self.blocks17(x)

        aux_out = None
        if self.training and self.aux_logits:
            aux_out = self.aux(x)

        # Reduction B
        x = self.reduction_b(x)

        # Block8 C组
        x = self.blocks8(x)

        # 分类输出
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        main_out = self.fc(x)

        if self.training and self.aux_logits:
            return main_out, aux_out
        return main_out

# -------------------------- 测试维度 --------------------------
if __name__ == "__main__":
    # 标准输入 299x299
    model = InceptionResNetV2(num_classes=26, aux_logits=True)
    model.train()
    dummy = torch.randn(2, 3, 299, 299)
    logits, aux = model(dummy)
    print("主输出 shape:", logits.shape)    # [2,26]
    print("辅助输出 shape:", aux.shape)     # [2,26]

    model.eval()
    pred = model(dummy)
    print("推理输出 shape:", pred.shape)    # [2,26]