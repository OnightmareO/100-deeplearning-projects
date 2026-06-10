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
        [transforms.Resize((100,400)), #(height,width)
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
        # Resize images to 400x100 pixels
        transforms.Resize((100, 400)),
        # Convert images to PyTorch tensors
        transforms.ToTensor(),
        # Normalize images using the provided mean and std
        transforms.Normalize(mean, std) # value在[-1,1]之间
    ]  

    augmentation_tfs = [
        transforms.Resize((100, 400)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), shear=5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ] 

    # Compose the main transformations into a single pipeline
    main_transform = transforms.Compose(main_tfs)

    transform_with_augmentation = transforms.Compose(augmentation_tfs)

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
    # torch.utils.data.random_split方法是全局随机打乱切分，而不是按类别均衡分割（分层抽样）
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=g)

    train_dataset = SubsetDataset(train_dataset, transform=augmentation_transform)
    val_dataset = SubsetDataset(val_dataset, transform=main_transform)
    test_dataset = SubsetDataset(test_dataset, transform=main_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_loop(model, criterion, optimizer, train_loader, val_loader, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_char_total = 0
        train_char_correct = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device) 
            optimizer.zero_grad()
            outputs = model(images)

            if outputs.ndim == 3:
                # 输出形状 [B, seq_len, num_classes]，标签形状 [B, seq_len]
                # view(-1) = 把所有东西铺平，只保留最后一个维度[B, seq_len, num_classes]→[B*seq_len, num_classes]
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                # dim=-1 = 在最后一维（36 个类别）里找最大值的位置
                predicted = outputs.argmax(dim=-1)
            else:
                loss = criterion(outputs, labels)
                predicted = outputs.argmax(dim=1)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_total += labels.size(0) 
            train_correct += (predicted == labels).all(dim=1).sum().item()
            train_char_total += labels.numel()
            train_char_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_char_accuracy = 100 * train_char_correct / train_char_total

        # 在验证集上评估模型性能
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_char_total = 0
        val_char_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                if outputs.ndim == 3:
                    loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                    predicted = outputs.argmax(dim=-1)
                else:
                    loss = criterion(outputs, labels)
                    predicted = outputs.argmax(dim=1)

                val_loss += loss.item()
                val_total += labels.size(0) 
                val_correct += (predicted == labels).all(dim=1).sum().item()
                val_char_total += labels.numel()
                val_char_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_char_accuracy = 100 * val_char_correct / val_char_total

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        print(f'Train char Acc: {train_char_accuracy:.2f}%, val char Acc: {val_char_accuracy:.2f}%')
        # 返回最后一个 epoch 的验证准确率与验证损失，供外部调用（如 Optuna 目标函数）使用
    return val_accuracy, avg_val_loss
    
