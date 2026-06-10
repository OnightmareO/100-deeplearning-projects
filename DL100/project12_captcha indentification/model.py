import torch
import torch.nn as nn

class FlexibleCNN(nn.Module):
    '''
    该卷积神经网络的网络结构由给定超参数确定，可灵活设置卷积层层数。
    分类器（全连接层）在首次前向传播过程中完成构建，以此适配卷积特征提取模块的输出尺寸。
    '''
    def __init__(self, n_layers, n_filters, kernel_sizes, dropout_rate, fc_size):

        super(FlexibleCNN, self).__init__()
        blocks = []
        in_channels = 1  # 输入图像的通道数（RGB）

        for i in range(n_layers):

            out_channels = n_filters[i]
            kernel_size = kernel_sizes[i]
            # 为了保持卷积层输出尺寸不变，计算适当的 padding
            padding = (kernel_size - 1) // 2

            # 构建卷积块：卷积层 + BatchNorm + ReLU 激活 + 2x2 最大池化
            block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)

                )
            blocks.append(block)
            in_channels = out_channels  # 更新输入通道数为当前层的输出通

        self.features = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_rate = dropout_rate
        self.fc_size = fc_size

        self.classifier = None  # 分类器将在首次前向传播时构建
    
    def _create_classifier(self, flattened_size, device, seq_len, classes_per_pos):
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(flattened_size, self.fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fc_size, seq_len*classes_per_pos),  # Assumes num_classes output classes
            nn.Unflatten(1,(seq_len, classes_per_pos))
        ).to(device)
    
    def forward(self, x):
        device = x.device
        x = self.features(x)
        flattened = torch.flatten(x, 1)  # 展平卷积特征，保留 batch 维度
        flattened_size = flattened.size(1)
        if self.classifier is None:
            self._create_classifier(flattened_size, device, seq_len=5, classes_per_pos=36)

        return self.classifier(flattened) # 模型输出 [B, seq_len, num_classes]