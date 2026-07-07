import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from data import parse_voc_target, get_pos_neg_samples
from model import build_rcnn_model, extract_roi_feature
from utils import crop_roi,generate_box,nms,compute_iou,voc_11point_ap
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
import numpy as np
from torchvision.datasets import VOCDetection
from evaluate import evaluate_dataset
import joblib

# 第一部分：微调Alexnet
def finetune_cnn(dataset, epochs, device):
    optimizer = optim.SGD(list(feature_extractor.parameters())+list(fc_head.parameters()),
                      lr = 0.0001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(dataset,batch_size=1,shuffle=True)
    feature_extractor.train()
    fc_head.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for i,(img,target) in enumerate(train_loader):
            # (N, C, H, W) → (H, W, C) 
            img_np = img[0].permute(1,2,0).numpy()

            # 获取真实框
            gts = parse_voc_target(target)

            # 单张图提取正负样本
            pos, pos_cls, pos_gt, neg = get_pos_neg_samples(img_np, gts)
            all_roi = pos + neg
            labels = []
            for c in pos_cls:
                labels.append(cls2idx[c])
            # 若 num_classes==20 且有 3 个负样本，追加后为 [20, 20, 20]
            labels += [num_classes] * len(neg)
            if len(all_roi) == 0:
                continue

            loss_batch = 0.0
            valid_count = 0
            for box, label in zip(all_roi, labels):
                roi_tensor = crop_roi(img_np, box, device)
                if roi_tensor is None:
                    continue
                feat = feature_extractor(roi_tensor)
                output = fc_head(feat)
                loss = criterion(output, torch.tensor([label], device=device))
                if not torch.isfinite(loss).all():
                    continue
                loss_batch += loss
                valid_count += 1

            if valid_count == 0 or not torch.isfinite(loss_batch):
                continue

            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            total_loss += loss_batch.item()

        print(f"Epoch {epoch} Finished, Avg Loss: {total_loss/len(train_loader):.4f}")

    # 微调完成冻结CNN
    feature_extractor.eval()
    fc_head.eval()

    torch.save({
    "feature_extractor": feature_extractor.state_dict(),
    "fc_head": fc_head.state_dict()}, "./models/19.rcnn/alexnet_finetuned.pth")


# 阶段2：收集全部特征，训练SVM与边框回归
def collect_all_features(dataset):
    all_pos_feats = [] # 正样本的4096为特征值
    all_pos_cls = [] # 正样本的类别索引
    all_pos_box_prop = [] # 正样本的候选框坐标
    all_pos_box_gt = [] # 正样本的GT框坐标
    all_neg_feats = [] # 负样本的4096为特征值
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print("Collecting all ROI features ...")
    for i,(img,target) in enumerate(train_loader):
        # (N, C, H, W) → (H, W, C) 
        img_np = img[0].permute(1,2,0).numpy()
        # 获取真实框
        gts = parse_voc_target(target)
        # 单张图提取正负样本
        pos, pos_cls, pos_gt, neg = get_pos_neg_samples(img_np, gts)
        
        for box, c_name, gt_box in zip(pos, pos_cls, pos_gt):
            roi_tensor = crop_roi(img_np, box, device)
            feat = extract_roi_feature(roi_tensor, feature_extractor, fc_head)
            all_pos_feats.append(feat)
            all_pos_cls.append(c_name)
            all_pos_box_prop.append(box)
            all_pos_box_gt.append(gt_box)
        # 负样本特征
        for box in neg:
            roi_tensor = crop_roi(img_np, box, device)
            feat = extract_roi_feature(roi_tensor, feature_extractor, fc_head)
            all_neg_feats.append(feat)
    return all_pos_feats, all_pos_cls, all_pos_box_prop, all_pos_box_gt, all_neg_feats


def train_svm(all_pos_feats, all_pos_cls, all_neg_feats):
    """训练每个类别对应的线性 SVM（one-vs-all）。

    该函数针对 VOC 中的每一个类别训练一个独立的 LinearSVC。
    对于当前类别 c，正样本为所有属于 c 的 ROI 特征，其他类别的正样本和全部负样本都视为负样本。

    参数:
        all_pos_feats (list): 正样本特征列表，每个元素为 4096 维特征向量。
        all_pos_cls (list): 正样本类别名称列表，与 all_pos_feats 一一对应。
        all_neg_feats (list): 负样本特征列表，通常来自背景或非目标候选框。

    返回:
        list: 长度为 num_classes 的 LinearSVC 模型列表，按 VOC_CLASSES 顺序对应每个类别。
    """
    
    svms = [None for _ in range(num_classes)]
    for c_idx in range(num_classes):
        target_cls = VOC_CLASSES[c_idx]
        x = []
        y = []
        has_pos = False

        for feat, c_name in zip(all_pos_feats, all_pos_cls):
            x.append(feat)
            y.append(1 if c_name == target_cls else 0)
            if c_name == target_cls:
                has_pos = True

        for feat in all_neg_feats:
            x.append(feat)
            y.append(0)

        if not has_pos or len(set(y)) < 2:
            continue

        try:
            svm = LinearSVC(C=1.0)
            svm.fit(x, y)
            svms[c_idx] = svm
        except ValueError:
            svms[c_idx] = None

    return svms

# 训练阶段 3：每类边框回归器
def train_bbox_regressor(all_pos_feats,all_pos_cls,all_pos_box_prop, all_pos_box_gt):
    """为每个类别训练边框回归器（LinearRegression）。

    说明：采用 one-vs-class 的方式为每个目标类别训练一个单独的线性回归器，输出是对候选框到真实框的偏移量回归。

    输入编码 (proposal -> target)：
        - dx = (x1_gt - x1_prop) / w_prop
        - dy = (y1_gt - y1_prop) / h_prop
        - dw = log(w_gt / w_prop)
        - dh = log(h_gt / h_prop)

    参数:
        all_pos_feats (list): 正样本特征列表，元素为 4096 维向量（来自 `extract_roi_feature`）。
        all_pos_cls (list): 正样本对应的类别名称列表，与 all_pos_feats 对应。
        all_pos_box_prop (list): 正样本的候选框（proposal）坐标列表，元素格式为 (x1, y1, x2, y2)。
        all_pos_box_gt (list): 正样本的对应真实框（ground-truth）坐标列表，格式同上。

    返回:
        list: 长度为 `num_classes` 的回归模型列表（`LinearRegression`），按 `VOC_CLASSES` 顺序。

    注意:
        - 仅对与当前类别匹配的正样本（c_name == target_class）构造训练样本；
        - 若某类别没有正样本则跳过该类别的训练，保留未训练的回归器实例；
        - 输出 y 为形如 [dx, dy, dw, dh] 的四元组，用于在检测阶段恢复预测框。
    """
    regs = [LinearRegression() for _ in range(num_classes)]
    for i in range(num_classes):
        target_class = VOC_CLASSES[i]
        x = []
        y = []
        for feat, c_name, prop, gt in zip(all_pos_feats, all_pos_cls, all_pos_box_prop, all_pos_box_gt):
            if c_name != target_class:
                continue
            x1p, y1p, x2p, y2p = prop
            x1g, y1g, x2g, y2g = gt
            wp = x2p - x1p
            hp = y2p - y1p
            # 宽高对数缩放因子
            dw = np.log((x2g - x1g) / wp)
            dh = np.log((y2g - y1g) / hp)
            # 相对偏移比例
            dx = (x1g - x1p) / wp
            dy = (y1g - y1p) / hp
            x.append(feat)
            y.append([dx, dy, dw, dh])
        if len(x) > 0:
            regs[i].fit(x, y)
    return regs


if __name__ == "__main__":
    VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    cls2idx = {c: i for i, c in enumerate(VOC_CLASSES)}
    num_classes = len(VOC_CLASSES)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor, fc_head = build_rcnn_model(device)
    transform = transforms.ToTensor()
    train_data = VOCDetection(root='./data/VOC2012/', year='2012', image_set='trainval', download=False, transform=transform)
    test_data = VOCDetection(root='./data/VOC2012/', year='2007', image_set='test', download=False, transform=transform)
    #train_data = Subset(train_data, range(10))
    #test_data = Subset(test_data, range(10))
    print("finetune_cnn start...")
    finetune_cnn(train_data, epochs=2,device=device)
    pos_feats, pos_cls, pos_props, pos_gts, neg_feats = collect_all_features(train_data)
    print("training svm")
    svm_models = train_svm(pos_feats, pos_cls, neg_feats)
    print("training bbox regressor")
    bbox_models = train_bbox_regressor(pos_feats, pos_cls, pos_props, pos_gts)
    print("SVM & BBox Regressor train done.")

    joblib.dump(svm_models, "./models/19.rcnn/rcnn_svm.pkl")
    joblib.dump(bbox_models,"./models/19.rcnn/rcnn_regs.pkl")
    
    print("Saved SVM & BBox Regressor models.")


   

    


