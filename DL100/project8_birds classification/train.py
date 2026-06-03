import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from data import BirdDataset
from utils import get_mean_std, get_transformations,get_dataloader
from model import ResNet, Bottleneck
from earlystopping import EarlyStopping

bird_dataset = BirdDataset(root_dir='./data/bird_photos/', transform=None)
'''
#检查数据集
print(min(bird_data[0].size[0] for bird_data in bird_dataset))
print(min(bird_data[0].size[1] for bird_data in bird_dataset))
print(f'Length of the dataset: {len(bird_dataset)}')
sel_idx = 10
img, label = bird_dataset[sel_idx]
print(f'Label: {label}')
print(f'Description: {bird_dataset.get_label_description(label)}')
# Print its shape
print(f'Image shape: {img.size}')  # PIL image size is (width, height)
'''

mean, std = get_mean_std(bird_dataset)
main_transform, transform_with_augmentation = get_transformations(mean, std)
train_loader, val_loader, test_loader = get_dataloader(
    dataset=bird_dataset,
    batch_size=16,
    val_part=0.1,
    test_part=0.1,
    main_transform=main_transform,
    augmentation_transform=transform_with_augmentation,
)

'''
#检查数据加载器和数据集
train_dataset = train_loader.dataset
val_dataset = val_loader.dataset
test_dataset = test_loader.dataset

print('=== Train Loader ===')
print(f"Number of batches in train_loader: {len(train_loader)}")
print(f"Number of samples in train_dataset: {len(train_dataset)}")
print(f"Transforms applied to train_dataset: {train_dataset.transform}")
print(f"train_dataset type: {type(train_dataset)}")

print('\n=== Validation Loader ===')
print(f"Number of batches in val_loader: {len(val_loader)}")
print(f"Number of samples in val_dataset: {len(val_dataset)}")
print(f"Transforms applied to val_dataset: {val_dataset.transform}")
print(f"val_dataset type: {type(val_dataset)}")

print('\n=== Test Loader ===')
print(f"Number of batches in test_loader: {len(test_loader)}")
print(f"Number of samples in test_dataset: {len(test_dataset)}")
print(f"Transforms applied to test_dataset: {test_dataset.transform}")
print(f"test_dataset type: {type(test_dataset)}")

'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('./models', exist_ok=True)  
model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=4).to(device) #ResNet-50 的层数配置为 [3, 4, 6, 3]

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

early_stopper = EarlyStopping(patience=10, min_delta=0.001, verbose=True, path='./models/bird_best.pth', mode='min')
num_epochs = 20
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_accuracy = 100 * train_correct / train_total
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_accuracy = 100 * val_correct / val_total
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

    if early_stopper(avg_val_loss, model):
        print(f'Early stopping triggered at epoch {epoch+1}')
        break

print("Training finished. Loading best model from early stopping checkpoint.")
model.load_state_dict(torch.load('./models/bird_best.pth'))
print("Loaded best model from ./models/bird_best.pth")
torch.save(model.state_dict(), './models/8.bird_classifier.pth')
print("Saved best model to ./models/8.bird_classifier.pth")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: {100 * correct / total:.2f}%')
