import torch
from torch.utils.data import Dataset
from data import TrafficSignDataset
from utils import get_mean_std, get_transformations,get_dataloader
import torch.nn as nn
import torch.optim as optim
import os
from model import InceptionResNetV2
traffic_data = TrafficSignDataset('./data/14_traffic_sign')
mean, std = get_mean_std(traffic_data)
main_transformation, aug_transformation = get_transformations(mean,std)
train_loader, val_loader, test_loader = get_dataloader(
    dataset=traffic_data,
    batch_size=16,
    val_part=0.1,
    test_part=0.1,
    main_transformation=main_transformation,
    aug_transformation=aug_transformation
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
os.makedirs('./models/',exist_ok=True)
print('Load model ...')
model = InceptionResNetV2(num_classes=58, aux_logits=True, dropout_rate=0.3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

print('Start Training ...')
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
        logits, aux_logits = model(images)
        loss = criterion(logits, labels) + 0.3 * criterion(aux_logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        _, predicted = torch.max(logits.data, 1)
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
            logits = model(images)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            
            _, predicted = torch.max(logits.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_accuracy = 100 * val_correct / val_total
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

print('Train Finished...')
torch.save(model.state_dict(),'./models/14.traffic_identifier.pth')
print("Saved best model to ./models/14.traffic_identifier.pth")

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

