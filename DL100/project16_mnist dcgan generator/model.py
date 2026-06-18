import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        #  网络结构是纯串行 Sequential，不存在多分支复用输入,所以nn.ReLU(True)
        #  直接覆盖修改输入张量本身，不分配新内存
        self.main = nn.Sequential(
            # 这一层后面紧跟着 BatchNorm2d，BatchNorm 内部自带可学习偏移参数 bias，
            # 卷积层再加偏置属于重复冗余、浪费参数，所以 bias=False。
            #input :[batch, 100, 1, 1]
            # H_out = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding
            # W_out = (W_in - 1) * stride - 2 * padding + kernel_size + output_padding
            nn.ConvTranspose2d(in_channels=100, out_channels=512,kernel_size=4,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),  # [batch, 512, 4, 4]

            nn.ConvTranspose2d(512, 256, 4, 2, 1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True), # [batch, 256, 8, 8]

            nn.ConvTranspose2d(256, 128, 4, 2, 1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True), # [batch, 128, 16, 16]

            nn.ConvTranspose2d(128, 64, 4, 2, 1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True), # [batch, 64, 32, 32]
 
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh() # [batch, 1, 64, 64]
        )
    
    def forward(self,x):
        return self.main(x)
    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        # H_out = floor((H_in + 2*padding - kernel_size) / stride + 1)
        self.main = nn.Sequential(
            # input: [batch, 1, 64, 64]
            nn.Conv2d(1,64,4,2,1,bias= False),
            nn.LeakyReLU(0.2, inplace=True),
            # [batch, 64, 32, 32]

            nn.Conv2d(64,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True), # [batch, 128, 16, 16]

            nn.Conv2d(128,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True), # [batch, 256, 8, 8]


            nn.Conv2d(256,512,4,2,1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True), # [batch, 512, 4, 4]

            nn.Conv2d(512, 1, 4, 1, 0, bias=False) # [batch, 1, 1, 1]
        )

    def forward(self,x):
        return self.main(x)
    

def weight_init(m):
    # m是网络里的每一层实例对象
    classname = m.__class__.__name__
    # 找到Conv层
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    # 找到BatchNorm层
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



if __name__ == "__main__":
    # 创建生成器和判别器
    generator = Generator()
    discriminator = Discriminator()
    
    # 创建随机噪声输入（100维）
    batch_size = 32
    noise = torch.randn(batch_size, 100,1,1)
    
    # 生成器生成假图像
    fake_images = generator(noise)
    print(f"生成的假图像形状: {fake_images.shape}")  # [32, 1, 64, 64]
    
    # 判别器判别图像
    fake_output = discriminator(fake_images)
    print(f"判别器输出（假图像）形状: {fake_output.shape}")  # [32, 1, 1, 1]
    
    # 创建真实图像（示例）
    real_images = torch.randn(batch_size, 1, 64, 64)
    real_output = discriminator(real_images)
    print(f"判别器输出（真实图像）形状: {real_output.shape}")  # [32, 1, 1, 1]
    
    # 打印模型参数量
    print(f"\n生成器参数量: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"判别器参数量: {sum(p.numel() for p in discriminator.parameters()):,}")