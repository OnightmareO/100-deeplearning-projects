import torch
from utils import save_images,save_model,load_model
from model import Generator, Discriminator
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_images = 5

generator = Generator().to(device)
generator, _ = load_model(model=generator,model_dir='./models/',model_name='17.generator.pth')
generator.to(device)

noise = torch.randn(num_images,100,1,1, device=device)
with torch.no_grad():
    generated_images = generator(noise).cpu() #[num_images, 3, 64, 64]

fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
# 遍历生成的图像，并在子图中显示
for idx in range(num_images):
    image = generated_images[idx].squeeze()
    image = (image * 0.5 + 0.5) * 255
    image = torch.clamp(image, min=0.0, max=255.0)
    # 将生成图像从 CHW 转成 HWC
    image = image.permute(1, 2, 0).numpy().astype('uint8')
    axes[idx].imshow(image)
    # 关闭坐标轴
    axes[idx].axis('off')
# 调整子图布局
plt.tight_layout()
# 显示图像
plt.show()