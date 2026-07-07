import torchvision.models as models
import torch.nn as nn
import torch

def build_rcnn_model(device):
    # 加载预训练AlexNet
    alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(device)
    # 特征提取层
    feature_extractor = nn.Sequential(*list(alexnet.features.children())).to(device)
    # 分类头：输出21类(20目标+背景)，中间4096维特征给SVM/回归
    fc_head = nn.Sequential(
        nn.AdaptiveAvgPool2d((6,6)),
        nn.Flatten(),
        *list(alexnet.classifier[:-1]), #把最后一层去掉
        nn.Linear(4096, 20 + 1)
    ).to(device)
    return feature_extractor, fc_head

# 提取4096维特征（冻结后用于SVM/边框回归）
def extract_roi_feature(roi_tensor, feature_extractor, fc_head):
    with torch.no_grad():
        feat_map = feature_extractor(roi_tensor)
        feat_4096 = fc_head[:-1](feat_map) # 把最后一层去掉，提取4096维特征
    return feat_4096.detach().cpu().numpy().squeeze(0)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(device)
    print(alexnet)