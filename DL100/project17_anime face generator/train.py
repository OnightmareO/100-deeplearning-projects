import torch
import torchvision.transforms as transforms
import torch.nn as nn
from utils import save_images,save_model,load_model
from torchvision.datasets import MNIST
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Generator, Discriminator, weight_init
from data import AnimefaceDataset

torch.manual_seed(42)
 
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = AnimefaceDataset(root_dir='./data/anime face',transform = transform)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
'''
print(f"Number of batches in train_loader: {len(data_loader)}")
print(f"Number of samples in train_dataset: {len(data_loader.dataset)}")
print(f"Transforms applied to train_dataset: {data_loader.dataset.transform}")
print(f"train_dataset type: {type(data_loader.dataset)}")
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
discriminator = Discriminator().to(device)
discriminator.apply(weight_init)
generator = Generator().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer_g = optim.Adam(generator.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizer_d = optim.Adam(discriminator.parameters(),lr=0.0002,betas=(0.5,0.999))
epochs = 20
# 训练过程
g_losses, d_losses = [], []
with torch.no_grad():
    noise = torch.randn(25, 100, 1, 1, device=device)
    fake_images = generator(noise).cpu()
    save_images(fake_images, epoch=0, prefix='generated_before_training')


print('Start training')
for epoch in range(epochs):
    g_running_loss, d_running_loss = 0.0, 0.0
    for real_images in data_loader:
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        optimizer_d.zero_grad()
        real_labels = torch.ones(batch_size, 1,1,1).to(device)
        fake_labels = torch.zeros(batch_size, 1,1,1).to(device)

        outputs = discriminator(real_images)
        d_real_loss = criterion(outputs, real_labels)

        # 生成假图像并计算判别器对假图像的损失
        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        fake_images = generator(noise)
        # 在训练判别器（D）时，需阻止假图像上的梯度回传到生成器（G），否则会同时更新 G
        outputs = discriminator(fake_images.detach())  # detach()用于阻止梯度传播到生成器
        d_fake_loss = criterion(outputs, fake_labels)

        # 计算判别器总损失并执行反向传播和优化
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_d.step()
        d_running_loss += d_loss.item()

        optimizer_g.zero_grad()
        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        fake_images = generator(noise)
        outputs = discriminator(fake_images)
        # 如果在训练生成器（要更新 G）时，不能 detach()
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_g.step()
        g_running_loss += g_loss.item()

    g_losses.append(g_running_loss / len(data_loader))
    d_losses.append(d_running_loss / len(data_loader))

    print(f"Epoch {epoch+1}, Generator Loss: {g_losses[-1]}, Discriminator Loss: {d_losses[-1]}")
    # 每隔一定的epoch保存生成器和判别器的模型，并保存生成的图像和部分真实图像
    if (epoch+1) % 5 == 0:
        save_model(generator, epoch+1, model_name='17.generator.pth')
        save_model(discriminator, epoch+1, model_name='17.discriminator.pth')

        with torch.no_grad():
            noise = torch.randn(25, 100,1,1, device=device)
            fake_images = generator(noise).cpu()
            save_images(fake_images, epoch=epoch+1, prefix='generated')

        # 从该迭代器取出下一批真实图片（这里是第一批数据
        real_images = next(iter(data_loader))
        save_images(real_images[:25], epoch=epoch+1, prefix='real')


