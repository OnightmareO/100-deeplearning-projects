import torch
import torch.nn as nn
import torchvision.models as tv_models

# 微调：冻结预训练模型的卷积层权重，只训练最后的全连接层。
'''
由于分类器是一个Sequential 模块，你需要将序列中的最后一个元素（[-1]）替换为
你的 new_classifier,这样就将旧的 ImageNet 分类器替换为你的自定义分类器。
在微调过程中，这个新层将是唯一会学习的部分。
'''
def get_vgg19_model(num_classes=6):
    model = tv_models.vgg19_bn(weights='IMAGENET1K_V1')
    # print(model)
    for param in model.features.parameters():
        param.requires_grad = False
    original_last_layer = model.classifier[-1]
    print(f"Original classifier: {original_last_layer}")

    num_features = original_last_layer.in_features
    model.classifier[-1] = nn.Linear(num_features, num_classes)
    print("Model's New Fully Connected Layer:")
    #print(model)
    return model