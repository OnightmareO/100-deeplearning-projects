import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# InceptionA：5×5 → 两层 3×3
class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super().__init__()
        self.branch1x1 = ConvBlock(in_channels, 64, kernel_size=1)
        self.branch5x5 = nn.Sequential(
            ConvBlock(in_channels, 48, kernel_size=1),
            ConvBlock(48, 64, kernel_size=3, padding=1)
        )
        self.branch3x3dbl = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=1),
            ConvBlock(64, 96, kernel_size=3, padding=1),
            ConvBlock(96, 96, kernel_size=3, padding=1)
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, pool_features, kernel_size=1)
        )

    def forward(self, x):
        b1 = self.branch1x1(x)
        b2 = self.branch5x5(x)
        b3 = self.branch3x3dbl(x)
        b4 = self.branch_pool(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class ReductionA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch3x3 = ConvBlock(in_channels, 384, kernel_size=3, stride=2)
        self.branch3x3dbl = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=1),
            ConvBlock(64, 96, kernel_size=3, padding=1),
            ConvBlock(96, 96, kernel_size=3, stride=2)
        )
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        b1 = self.branch3x3(x)
        b2 = self.branch3x3dbl(x)
        b3 = self.branch_pool(x)
        return torch.cat([b1, b2, b3], dim=1)

# InceptionB：7×7 → (1×7)+(7×1) 长条卷积
class InceptionB(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.branch1x1 = ConvBlock(in_channels, 192, kernel_size=1)
        self.branch7x7 = nn.Sequential(
            ConvBlock(in_channels, hidden_channels, kernel_size=1),
            ConvBlock(hidden_channels, hidden_channels, kernel_size=(1, 7), padding=(0, 3)),
            ConvBlock(hidden_channels, 192, kernel_size=(7, 1), padding=(3, 0))
        )
        self.branch7x7dbl = nn.Sequential(
            ConvBlock(in_channels, hidden_channels, kernel_size=1),
            ConvBlock(hidden_channels, hidden_channels, kernel_size=(1, 7), padding=(0, 3)),
            ConvBlock(hidden_channels, hidden_channels, kernel_size=(7, 1), padding=(3, 0)),
            ConvBlock(hidden_channels, hidden_channels, kernel_size=(1, 7), padding=(0, 3)),
            ConvBlock(hidden_channels, 192, kernel_size=(7, 1), padding=(3, 0))
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, 192, kernel_size=1)
        )

    def forward(self, x):
        b1 = self.branch1x1(x)
        b2 = self.branch7x7(x)
        b3 = self.branch7x7dbl(x)
        b4 = self.branch_pool(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class ReductionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch3x3 = nn.Sequential(
            ConvBlock(in_channels, 192, kernel_size=1),
            ConvBlock(192, 320, kernel_size=3, stride=2)
        )
        self.branch7x7x3 = nn.Sequential(
            ConvBlock(in_channels, 192, kernel_size=1),
            ConvBlock(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            ConvBlock(192, 192, kernel_size=(7, 1), padding=(3, 0)),
            ConvBlock(192, 192, kernel_size=3, stride=2)
        )
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        b1 = self.branch3x3(x)
        b2 = self.branch7x7x3(x)
        b3 = self.branch_pool(x)
        return torch.cat([b1, b2, b3], dim=1)

# InceptionC：3×3 拆成并行 1×3/3×1，大幅减少参数量
class InceptionC(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1x1 = ConvBlock(in_channels, 320, kernel_size=1)
        self.branch3x3 = ConvBlock(in_channels, 384, kernel_size=1)
        self.branch3x3_a = ConvBlock(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_b = ConvBlock(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3dbl = nn.Sequential(
            ConvBlock(in_channels, 448, kernel_size=1),
            ConvBlock(448, 384, kernel_size=3, padding=1)
        )
        self.branch3x3dbl_a = ConvBlock(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_b = ConvBlock(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, 192, kernel_size=1)
        )

    def forward(self, x):
        b1 = self.branch1x1(x)
        b2 = self.branch3x3(x)
        b2 = torch.cat([self.branch3x3_a(b2), self.branch3x3_b(b2)], dim=1)
        b3 = self.branch3x3dbl(x)
        b3 = torch.cat([self.branch3x3dbl_a(b3), self.branch3x3dbl_b(b3)], dim=1)
        b4 = self.branch_pool(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class AuxLogits(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # 固定输出为 4×4
        self.conv = ConvBlock(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ---------------------- 适配224×224的主模型（仅头部改动） ----------------------
class InceptionV3_224(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, dropout=0.2):
        super().__init__()
        self.aux_logits = aux_logits

        # ========== 关键改动1：第一层卷积stride从2→1，padding调整 ==========
        self.Conv2d_1a_3x3 = ConvBlock(3, 32, kernel_size=3, stride=1, padding=1)
        self.Conv2d_2a_3x3 = ConvBlock(32, 32, kernel_size=3, padding=1)
        self.Conv2d_2b_3x3 = ConvBlock(32, 64, kernel_size=3, padding=1)
        
        # ========== 关键改动2：第一个maxpool padding=1，保证尺寸对半切 ==========
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.Conv2d_3b_1x1 = ConvBlock(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = ConvBlock(80, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 下面所有Inception堆叠完全和论文原版一致，通道不变
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)

        self.Mixed_6a = ReductionA(288)

        self.Mixed_6b = InceptionB(768, hidden_channels=128)
        self.Mixed_6c = InceptionB(768, hidden_channels=160)
        self.Mixed_6d = InceptionB(768, hidden_channels=160)
        self.Mixed_6e = InceptionB(768, hidden_channels=192)

        if aux_logits:
            self.AuxLogits = AuxLogits(768, num_classes)

        self.Mixed_7a = ReductionB(768)
        self.Mixed_7b = InceptionC(1280)
        self.Mixed_7c = InceptionC(2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)

        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)

        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)

        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)

        aux = None
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)

        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        out = self.fc(x)

        if self.aux_logits and self.training:
            return out, aux
        return out


# 尺寸测试
if __name__ == "__main__":
    model = InceptionV3_224(num_classes=26, aux_logits=True)
    model.train()
    # 输入改为224×224
    dummy = torch.randn(4, 3, 224, 224)
    logits, aux_logits = model(dummy)
    print("主输出shape:", logits.shape)    # torch.Size([4, 26])
    print("辅助输出shape:", aux_logits.shape) # torch.Size([4, 26])

    model.eval()
    test_out = model(dummy)
    print("推理输出shape:", test_out.shape)