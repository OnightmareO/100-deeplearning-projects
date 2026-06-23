import torch
import torch.nn as nn

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        # Depthwise 逐通道卷积
        self.depthwise = nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size,stride=stride,
                                   padding=padding,groups=in_channels,bias=bias)
         # Pointwise 1×1 通道融合
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self,x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Xception 残差块（Middle Flow 重复使用）
class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reps, stride=1, start_with_relu=True, grow_first=True):
        super().__init__()
        layers = []
        channel = in_channels

        # 分支1：grow_first=True（先扩通道）
        if grow_first:
            layers.append(nn.ReLU())
             # 第一层可分离卷积直接把通道升到 out_channels
            layers.append(SeparableConv2d(in_channels,out_channels,padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            channel = out_channels

        # 重复堆叠可分离卷积
        for i in range(reps - 1):
            layers.append(nn.ReLU())
            layers.append(SeparableConv2d(channel, channel, padding=1))
            layers.append(nn.BatchNorm2d(channel))


        if not grow_first: #Exit Flow 第一个下采样块（唯一 False）
            # 最后一层才升通道到 out_channels
            layers.append(nn.ReLU())
            layers.append(SeparableConv2d(channel, out_channels, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
        
        # 深度可分离卷积 SeparableConv2d (stride=2) 完成尺寸减半，全程无任何 MaxPool、AvgPool 下采样层
        if stride != 1:
            layers.append(nn.ReLU())
            layers.append(SeparableConv2d(out_channels, out_channels, stride=stride, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
        
        self.block = nn.Sequential(*layers)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.start_with_relu = start_with_relu

    def forward(self, x):
        residual = self.shortcut(x)
        if self.start_with_relu:
            x = self.block(x)
        else:
            # 跳过第一层relu
            x = self.block[1:](x)
        x = x + residual
        return x
        
# Xception 主网络
class Xception(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # ---------------- Entry Flow 入口流 ----------------
        self.entry_flow = nn.Sequential(
            # 普通卷积层
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Block1
            XceptionBlock(64, 128, reps=2, stride=2, start_with_relu=False, grow_first=True),
            # Block2
            XceptionBlock(128, 256, reps=2, stride=2, start_with_relu=True, grow_first=True),
            # Block3
            XceptionBlock(256, 728, reps=2, stride=2, start_with_relu=True, grow_first=True),
        )

        # ---------------- Middle Flow 中间流（8个重复残差块） ----------------
        middle_blocks = []
        for _ in range(8):
            middle_blocks.append(
                XceptionBlock(728, 728, reps=3, stride=1, start_with_relu=True, grow_first=True)
            )
        self.middle_flow = nn.Sequential(*middle_blocks)

        # ---------------- Exit Flow 出口流 ----------------
        self.exit_flow = nn.Sequential(
            XceptionBlock(728, 1024, reps=2, stride=2, start_with_relu=True, grow_first=False),
            SeparableConv2d(1024, 1536, padding=1),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            SeparableConv2d(1536, 2048, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )

        # 分类头
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # 输入 x: [B,3,299,299] 原图输入尺寸299×299
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out
    
if __name__ == "__main__":
    model = Xception(num_classes=10)
    # Xception 标准输入尺寸 299×299
    dummy_input = torch.randn(2, 3, 299, 299)
    output = model(dummy_input)
    print("输出shape:", output.shape)  # [2, 10]