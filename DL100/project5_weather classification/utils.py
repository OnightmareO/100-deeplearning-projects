from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from data import SubsetDataset

def get_mean_std(dataset:Dataset):
    '''
    Computes the mean and standard deviation of a dataset.
    Args:
        dataset (Dataset): The dataset for which to compute the mean and standard deviation.
    Returns:
        mean (list): A list of mean values for each channel.
        std (list): A list of standard deviation values for each channel.
    '''
    preprocess = transforms.Compose(
        [transforms.Resize((256, 256)), 
         transforms.CenterCrop(224),
         transforms.ToTensor()]
    )
    means = []
    stds = []
    for img, _ in dataset:
        img = preprocess(img)
        # 遍历数据集中的每张图像，应用变换并在空间维度（高/宽）上使用 `dim=[1, 2]` 计算均值与标准差。
        means.append(img.mean(dim=(1, 2)))
        stds.append(img.std(dim=(1, 2)))
    # 将所有每图统计堆叠为张量，并对所有图像求平均以得到数据集级别的通道统计。
    mean = torch.stack(means).mean(dim=0)
    std = torch.stack(stds).mean(dim=0)
    return mean.tolist(), std.tolist()

def get_transformations(mean, std):
    '''
    Args:
        mean (list): A list of mean values for each channel.
        std (list): A list of standard deviation values for each channel.
    Returns:
        main_transform (transforms.Compose): A composition of transformations for the main dataset.
        transform_with_augmentation (transforms.Compose): A composition of transformations that includes data augmentation.
    '''
    main_tfs = [  
        # Resize images to 256x256 pixels
        transforms.Resize((256, 256)),
        # Center crop images to 224x224 pixels
        transforms.CenterCrop(224),
        # Convert images to PyTorch tensors
        transforms.ToTensor(),
        # Normalize images using the provided mean and std
        transforms.Normalize(mean, std)
    ]  

    augmentation_tfs = [  
        # Randomly flip the image vertically
        transforms.RandomVerticalFlip(p=0.5),
        # Randomly rotate the image by ±15 degrees
        transforms.RandomRotation(degrees=15)
    ]  

    # Compose the main transformations into a single pipeline
    main_transform = transforms.Compose(main_tfs)

    transform_with_augmentation = transforms.Compose(augmentation_tfs + main_tfs)

    return main_transform, transform_with_augmentation


g = torch.Generator()
g.manual_seed(42)
def get_dataloader(dataset, batch_size,val_part,test_part,main_transform,augmentation_transform):
    '''
    Args:
        dataset (Dataset): The dataset to be loaded.
        batch_size (int): The number of samples per batch.
        val_part (float): The proportion of the dataset to be used for validation.
        test_part (float): The proportion of the dataset to be used for testing.
        main_transform: Transform to apply to validation and test splits.
        augmentation_transform: Transform to apply to the training split.

    Returns:
        dataloader (DataLoader): A DataLoader for the given dataset.
    '''
    # 计算训练、验证和测试集的大小
    total_size = len(dataset)
    val_size = int(total_size * val_part)
    test_size = int(total_size * test_part)
    train_size = total_size - val_size - test_size

    # 使用 torch.utils.data.random_split 将数据集划分为训练、验证和测试子集
    # 这种划分方法是全局随机打乱切分，而不是按类别均衡分割（分层抽样）
    train_dataset, val_dataset, test_dataset = stratified_split(dataset, val_part, test_part, generator=g)

    train_dataset = SubsetDataset(train_dataset, transform=augmentation_transform)
    val_dataset = SubsetDataset(val_dataset, transform=main_transform)
    test_dataset = SubsetDataset(test_dataset, transform=main_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


#分层抽样函数
def stratified_split(dataset, val_ratio, test_ratio, generator=None):
    '''
    Args:
        dataset (Dataset): The dataset to be split.
        val_ratio (float): The proportion of the dataset to be used for validation.
        test_ratio (float): The proportion of the dataset to be used for testing.
        generator: A random generator for reproducibility.
    Returns:
        train_ds (Subset): The training subset of the dataset.
        val_ds (Subset): The validation subset of the dataset.
        test_ds (Subset): The test subset of the dataset.
    '''
    # 按标签分组索引（关键：把同一类的图片索引放一起）
    label_indices = defaultdict(list)
    for idx, label in enumerate(dataset.labels):
        label_indices[label].append(idx)

    train_indices = []
    val_indices = []
    test_indices = []

    # 对每一类，独立切分 8:1:1
    for label, indices in label_indices.items():
        # 生成一个随机排列的序号
        perm = torch.randperm(len(indices), generator=generator).tolist()
        # 根据随机顺序，重新排列 indices
        indices = [indices[i] for i in perm]
        total = len(indices)
        
        val_size = int(total * val_ratio)
        test_size = int(total * test_ratio)
        train_size = total - val_size - test_size

        # 分配索引
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size+val_size]
        test_idx = indices[train_size+val_size:]

        # 将分配的索引添加到对应的列表中，一定要 extend 而不是 append
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
        test_indices.extend(test_idx)

    # 生成子集
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    test_ds = Subset(dataset, test_indices)
    return train_ds, val_ds, test_ds