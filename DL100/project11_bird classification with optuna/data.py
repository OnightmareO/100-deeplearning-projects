import os   
from torch.utils.data import Dataset
from PIL import Image

class BirdDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # 初始化根目录路径与数据变换操作
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = [] # 获取类别名称列表并排序
        # 遍历根目录下的所有子目录（每个子目录代表一个类别）
        for label, class_dir in enumerate(os.listdir(self.root_dir)):
            class_path = os.path.join(self.root_dir, class_dir)
            self.class_names.append(class_dir)  # 添加类别名称
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)
    
    def get_label_description(self, label: int):
        """
        Returns the description of a class label.
        """
        description = self.class_names[label]
        return description

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    
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
