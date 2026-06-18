import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def save_model(model,epoch,save_dir= './models/',model_name='16dcgan_mnist.pth'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir,model_name) 
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, save_path)
    print(f"Model saved at: {save_path}")

def load_model(model, model_dir='./models/', model_name='16dcgan_mnist.pth'):
    model_path = os.path.join(model_dir,model_name) 
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from: {model_path}")
    return model, checkpoint['epoch'] 

def save_images(images, epoch, folder='./data/16_dcgan_generated/', prefix='generated'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    # 把一组图像拼成一张网格图像,nrow=5每行放 5 张图像
    grid = torchvision.utils.make_grid(images, nrow=5, normalize=True)
    # 构建图像保存的完整路径
    file_path = os.path.join(folder, f'{prefix}_epoch_{epoch}.png')
    # 使用torchvision.utils.save_image保存图像到文件
    torchvision.utils.save_image(grid, file_path)
    print(f"Images saved at: {file_path}")

def visualize_generated_images(generator,device, num_images=5):
    noise = torch.randn(num_images,generator,device=device)
    with torch.no_grad():
        generated_images = generator(noise).cpu()

    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    # 遍历生成的图像，并在子图中显示
    for idx in range(num_images):
        # 显示图像，并设置灰度色彩映射
        axes[idx].imshow(generated_images[idx].squeeze(), cmap='gray')
        # 关闭坐标轴
        axes[idx].axis('off')
    # 调整子图布局
    plt.tight_layout()
    # 显示图像
    plt.show()
