import torch
from torch.utils.data import Dataset, Subset
import os
from PIL import Image
import glob
import matplotlib.pyplot as plt


class CaptchaDataset(Dataset):
    def __init__(self, root_dir, transform=None, extensions=['png', 'jpg', 'jpeg', 'bmp', 'gif']):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.label_names = []
        self.label = []
        self.number = ['0','1','2','3','4','5','6','7','8','9']
        self.alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n',
                         'o','p','q','r','s','t','u','v','w','x','y','z']
        self.char_set =  {c:i for i,c in enumerate(self.number+self.alphabet)}

        # 收集指定扩展名的图片文件
        for ext in extensions:
            # os.path.join(root_dir, f'*.{ext}')：构建一个匹配指定扩展名的文件路径模式，例如 'data/captcha_images/*.png'
            # glob.glob(...)：返回匹配该模式的所有文件路径的列表（非递归），例如 ['images/a.png', 'images/b.png']
            self.image_paths.extend(glob.glob(os.path.join(root_dir, f'*.{ext}')))

        # 将文件名（不含扩展）作为标签
        for p in self.image_paths:
            #os.path.basename(p)：从路径 p 中取出文件名（含后缀），例如 'a.png'
            #os.path.splitext(...)：将文件名分割成两部分，返回一个元组 (root, ext)，其中 root 是文件名（不含扩展），ext 是扩展名，例如 ('a', '.png')
            label = os.path.splitext(os.path.basename(p))[0]
            self.label_names.append(label)
            self.label.append(self.text2vec(label))
            
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('L')
        if self.transform:
            img = self.transform(img)
        label = self.label[idx]

        return img, torch.tensor(label)
    
    # 返回原始标签字符串（文件名，不含扩展名）
    def get_label_str(self, idx):
        return self.label_names[idx]
    
    def text2vec(self,text:str):
        vector = []
        for i, c in enumerate(text):
            vector.append(self.char_set[c])
        return vector

class SubsetDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    data = CaptchaDataset(root_dir='./data/captcha_images_v2/',extensions=['png'])

    print(min(captcha_data[0].size[0] for captcha_data in data))
    print(min(captcha_data[0].size[1] for captcha_data in data))
    print(f'Length of the dataset: {len(data)}')
    sel_idx = 0
    img, label = data[sel_idx]
    print(f'Label: {label}')
    print(f'Description: {data.get_label_str(sel_idx)}')
    # Print its shape
    print(f'Image shape: {img.size}')  # PIL image size is (width, height)
  
    # 显示多张图像
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(min(6, len(data))):
        img, label = data[i]
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Label: {data.get_label_str(i)}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()



       