import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import VOCDetection
from torch.utils.data import Subset,DataLoader
from data import parse_voc_target
from model import build_rcnn_model, extract_roi_feature
from utils import compute_iou, crop_roi, generate_box, nms, voc_11point_ap


VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
num_classes = len(VOC_CLASSES)


def load_model_components(alexnet_path, svm_path, regs_path, device):
    feature_extractor, fc_head = build_rcnn_model(device)
    feature_extractor.eval()
    fc_head.eval()

    if alexnet_path.exists():
        checkpoint = torch.load(alexnet_path, map_location=device)
        if isinstance(checkpoint, dict) and 'feature_extractor' in checkpoint and 'fc_head' in checkpoint:
            feature_extractor.load_state_dict(checkpoint['feature_extractor'])
            fc_head.load_state_dict(checkpoint['fc_head'])
        else:
            feature_extractor.load_state_dict(checkpoint)
    else:
        print(f"AlexNet checkpoint not found: {alexnet_path}")
    
    print(f"Load AlexNet model from: {alexnet_path}")

    if svm_path.exists():
        svms = joblib.load(svm_path)
    else:
        raise FileNotFoundError(f"SVM file not found: {svm_path}")
    
    print(f"Load SVM models from: {svm_path}")

    if regs_path.exists():
        regs = joblib.load(regs_path)
    else:
        raise FileNotFoundError(f"Regression file not found: {regs_path}")
    
    print(f"Load Regression models from: {regs_path}")

    return feature_extractor, fc_head, svms, regs


def detect_with_svm_and_regressor(img, svms, regs, feature_extractor, fc_head, thres=0.6):
    """使用训练好的 SVM 分类器和回归器在单张图像上检测目标。

    处理流程：
        1. 使用 `generate_box` 生成候选框（proposals）
        2. 对每个 proposal 裁剪 ROI、提取特征；
        3. 使用每个类别的 SVM（`decision_function`）计算置信分数并阈值筛选；
        4. 对通过阈值的类别，使用对应的回归器预测偏移量 `[dx, dy, dw, dh]`，将 proposal 解码为预测框；
        5. 收集所有预测框、分数和类别索引，最后使用 NMS 去重并返回结果。

    参数:
        img (ndarray): 原始图像（H, W, C），用于生成 proposals 与裁剪 ROI。
        svms (list): 长度为 `num_classes` 的 LinearSVC 列表，每个元素用于该类别的 one-vs-all 判别。
        regs (list): 长度为 `num_classes` 的回归器列表（如 `LinearRegression`），用于预测边框偏移。
        thres (float): SVM 分数阈值，低于该阈值的候选框将被丢弃。

    返回:
        tuple: `(res_box, res_cls, res_score)`
            - res_box: 保留下来的预测框列表，每个框为 [x1, y1, x2, y2]。
            - res_cls: 对应的类别名称列表（VOC_CLASSES 中的名称）。
            - res_score: 对应的置信分数列表。

    说明与注意事项:
        - SVM 使用 `decision_function` 来获取连续分数，不能用 `predict`；
        - 回归器预测的是相对于 proposal 的偏移量，函数内用如下公式解码：
            x1 = x1_p + wp * dx
            y1 = y1_p + hp * dy
            w  = wp * exp(dw)
            h  = hp * exp(dh)
            x2 = x1 + w,  y2 = y1 + h
        - 最后调用 `nms`（IoU 阈值 0.3）去除重叠框；
        - 该函数依赖全局变量 `feature_extractor` 与 `fc_head` 来提取特征。
    """
    proposals = generate_box(img, topk=2000)
    out_boxes = []
    out_scores = []
    out_cls_idx = []

    for prop in proposals:
        roi_tensor = crop_roi(img, prop, device)
        if roi_tensor is None:
            continue
        feat = extract_roi_feature(roi_tensor, feature_extractor, fc_head)
        x1p, y1p, x2p, y2p = prop
        wp = x2p - x1p
        hp = y2p - y1p

        for c_idx in range(num_classes):
            if svms[c_idx] is None or regs[c_idx] is None:
                continue
            score = svms[c_idx].decision_function([feat])[0]
            if score < thres:
                continue

            dx, dy, dw, dh = regs[c_idx].predict([feat])[0]
            x1 = x1p + wp * dx
            y1 = y1p + hp * dy
            w = wp * np.exp(dw)
            h = hp * np.exp(dh)
            x2 = x1 + w
            y2 = y1 + h

            out_boxes.append([x1, y1, x2, y2])
            out_scores.append(score)
            out_cls_idx.append(c_idx)

    keep = nms(out_boxes, out_scores, thres=0.3)
    res_box = [out_boxes[i] for i in keep]
    res_cls = [VOC_CLASSES[out_cls_idx[i]] for i in keep]
    res_score = [out_scores[i] for i in keep]
    return res_box, res_cls, res_score


def evaluate_dataset(test_data, svms, regs, feature_extractor, fc_head, svm_thres=0.6, iou_thres=0.5):
    class_pred_records = {cls: [] for cls in VOC_CLASSES}
    class_gt_total = {cls: 0 for cls in VOC_CLASSES}

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    for img_tensor, target in test_loader:
        img_np = img_tensor[0].permute(1, 2, 0).numpy()
        gts = parse_voc_target(target)

        gt_dict = {}
        for gt in gts:
            cls_name = gt['cls']
            class_gt_total[cls_name] += 1
            gt_dict.setdefault(cls_name, []).append(gt['box'])

        pred_boxes, pred_cls_names, pred_scores = detect_with_svm_and_regressor(
            img_np, svms, regs, feature_extractor, fc_head, thres=svm_thres
        )

        pred_pack = list(zip(pred_scores, pred_cls_names, pred_boxes))
        pred_pack.sort(key=lambda x: x[0], reverse=True)

        matched_gt_idx = {c: set() for c in VOC_CLASSES}

        for score, cls_name, pred_box in pred_pack:
            if cls_name not in gt_dict or len(gt_dict[cls_name]) == 0:
                class_pred_records[cls_name].append((score, 0))
                continue

            gt_box_list = gt_dict[cls_name]
            max_iou = 0.0
            match_idx = -1
            for idx, gt_box in enumerate(gt_box_list):
                iou = compute_iou(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    match_idx = idx

            if max_iou >= iou_thres and match_idx not in matched_gt_idx[cls_name]:
                class_pred_records[cls_name].append((score, 1))
                matched_gt_idx[cls_name].add(match_idx)
            else:
                class_pred_records[cls_name].append((score, 0))

    class_ap_dict = {}
    all_ap = []
    for cls in VOC_CLASSES:
        records = class_pred_records[cls]
        total_gt = class_gt_total[cls]
        if total_gt == 0 or len(records) == 0:
            ap = 0.0
        else:
            records.sort(key=lambda x: x[0], reverse=True)
            tp_acc = 0
            fp_acc = 0
            prec_list = []
            rec_list = []
            for _, is_tp in records:
                if is_tp:
                    tp_acc += 1
                else:
                    fp_acc += 1
                precision = tp_acc / (tp_acc + fp_acc)
                recall = tp_acc / total_gt
                prec_list.append(precision)
                rec_list.append(recall)
            ap = voc_11point_ap(prec_list, rec_list)

        class_ap_dict[cls] = ap
        all_ap.append(ap)
        print(f"{cls:12s} AP = {ap:.4f}")

    mAP = float(np.mean(all_ap))
    print('-' * 30)
    print(f"mAP@IoU={iou_thres} = {mAP:.4f}")
    return mAP, class_ap_dict


def main():
    parser = argparse.ArgumentParser(description='Evaluate an RCNN-style detector with AlexNet, SVMs and bbox regressors')
    parser.add_argument('--alexnet-path', type=str, default=str(Path(__file__).resolve().parent.parent / 'models' / '19.rcnn' / 'alexnet_finetuned.pth'))
    parser.add_argument('--svm-path', type=str, default=str(Path(__file__).resolve().parent.parent / 'models' / '19.rcnn' / 'rcnn_svm.pkl'))
    parser.add_argument('--regs-path', type=str, default=str(Path(__file__).resolve().parent.parent / 'models' / '19.rcnn' / 'rcnn_regs.pkl'))
    parser.add_argument('--data-root', type=str, default=str(Path(__file__).resolve().parent.parent / 'data' / 'VOC2012'))
    parser.add_argument('--year', type=str, default='2007')
    parser.add_argument('--image-set', type=str, default='test')
    parser.add_argument('--svm-thres', type=float, default=0.6)
    parser.add_argument('--iou-thres', type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor, fc_head, svms, regs = load_model_components(
        Path(args.alexnet_path), Path(args.svm_path), Path(args.regs_path), device
    )

    transform = transforms.ToTensor()
    test_data = VOCDetection(root=args.data_root, year=args.year, image_set=args.image_set, download=False, transform=transform)
    #test_data = Subset(test_data,range(10))
    evaluate_dataset(test_data, svms, regs, feature_extractor, fc_head, svm_thres=args.svm_thres, iou_thres=args.iou_thres)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
