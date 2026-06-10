import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_mean_std, get_transformations, get_dataloader,train_loop
from model import FlexibleCNN
from data import CaptchaDataset, SubsetDataset
import os
from matplotlib import pyplot as plt

def objective(trial, device):
    """
    定义基于Optuna进行超参优化所用的目标函数。

    函数在每次试验中采样一组超参数、搭建模型、以固定轮数训练模型，
    在验证集上评估模型性能并返回准确率。Optuna依靠返回的准确率,
    引导算法搜寻最优超参数组合。

    参数：
        trial: Optuna的Trial对象,用于超参数采样。
        device: 模型训练与评估所用硬件设备(CPU或CUDA)。

    返回值：
        浮点型数值，代表训练完成后模型的验证集准确率。
    """

    # 采样超参数
    n_layers = trial.suggest_int('n_layers', 4, 6)  # 卷积层数
    n_filters = [trial.suggest_int(f'n_filters_{i}', 64, 512) for i in range(n_layers)]  # 每层卷积核数量
    kernel_sizes = [trial.suggest_categorical(f'kernel_size_{i}', [2, 4]) for i in range(n_layers)]  # 卷积核大小
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.3)  # Dropout率
    fc_size = trial.suggest_int('fc_size',2048,4096)  # 全连接层大小

    # 构建模型
    model = FlexibleCNN(n_layers, n_filters, kernel_sizes, dropout_rate, fc_size).to(device)
    captcha_dataset = CaptchaDataset(root_dir='./data/captcha_images_v2/', transform=None)

    mean, std = get_mean_std(captcha_dataset)
    main_transform, transform_with_augmentation = get_transformations(mean, std)
    train_loader, val_loader, _ = get_dataloader(
    dataset=captcha_dataset,
    batch_size=16,
    val_part=0.1,
    test_part=0.1,
    main_transform=main_transform,
    augmentation_transform=transform_with_augmentation,
    )
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 1e-4)

    # 固定训练轮数，训练模型并评估验证集性能
    val_accuracy, _ = train_loop(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=10)
    return val_accuracy  # 返回验证集准确率作为优化目标


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('./models', exist_ok=True)  

study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, device), n_trials=10) 
df = study.trials_dataframe()

best_trial = study.best_trial

print("Best trial:")
print(f" Value (Accuracy): {best_trial.value:.4f}")
print(" Hyperparameters:")
print(best_trial.params)

# 可视化优化历史
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.title('Optimization History')
plt.show()

# 可视化参数重要性
optuna.visualization.matplotlib.plot_param_importances(study)
plt.show()

# 可视化超参数之间的关系
ax = optuna.visualization.matplotlib.plot_parallel_coordinate(
    study, params=['n_layers', 'n_filters_0', 'kernel_size_0', 'dropout_rate', 'fc_size']
)
fig = ax.figure
fig.set_size_inches(12, 6, forward=True)  # forward=True updates the canvas
fig.tight_layout()



