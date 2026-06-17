import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()

        self.main = nn.Sequential(
            nn.Linear(in_features=100, out_features=256),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            nn.BatchNorm1d(256, momentum=0.8),

            nn.Linear(in_features=256, out_features=512),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            nn.BatchNorm1d(512, momentum=0.8),

            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            nn.BatchNorm1d(1024, momentum=0.8),

            nn.Linear(in_features=1024, out_features=28*28),
            nn.Tanh()
        )
    
    def forward(self,x):
        # 把这个张量重新变形为形状 [batch_size, 1, 28, 28]
        return self.main(x).view(-1, 1, 28, 28)
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=28*28, out_features=1024),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        return self.main(x)


if __name__ == "__main__":
    # 创建生成器和判别器
    generator = Generator()
    discriminator = Discriminator()
    
    # 创建随机噪声输入（100维）
    batch_size = 32
    noise = torch.randn(batch_size, 100)
    
    # 生成器生成假图像
    fake_images = generator(noise)
    print(f"生成的假图像形状: {fake_images.shape}")  # [32, 1, 28, 28]
    
    # 判别器判别图像
    fake_output = discriminator(fake_images)
    print(f"判别器输出（假图像）形状: {fake_output.shape}")  # [32, 1]
    
    # 创建真实图像（示例）
    real_images = torch.randn(batch_size, 1, 28, 28)
    real_output = discriminator(real_images)
    print(f"判别器输出（真实图像）形状: {real_output.shape}")  # [32, 1]
    
    # 打印模型参数量
    print(f"\n生成器参数量: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"判别器参数量: {sum(p.numel() for p in discriminator.parameters()):,}")
    

