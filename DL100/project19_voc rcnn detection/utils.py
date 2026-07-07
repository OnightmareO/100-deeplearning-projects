from PIL import Image, ImageDraw
import numpy as np
from skimage.segmentation import felzenszwalb
import torchvision.transforms as transform
def show_voc_image_with_boxes(dataset,index):
    img,target = dataset[index]

    # 将 PIL 图片转成可绘制对象
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)
    for obj in target['annotation']['object']: #有多个目标
        bbox = obj["bndbox"]
        x1 = int(bbox["xmin"])
        y1 = int(bbox["ymin"])
        x2 = int(bbox["xmax"])
        y2 = int(bbox["ymax"])
    
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, max(0, y1 - 10)), obj["name"], fill="red")
    
    draw_img.show()

def compute_iou(box1,box2):
    """
    计算两个矩形框的交并比（IoU）。

    参数:
        box1 (tuple): 第一个矩形框，格式为 (x1, y1, x2, y2)，表示左上角和右下角坐标。
        box2 (tuple): 第二个矩形框，格式为 (x1, y1, x2, y2)，表示左上角和右下角坐标。

    返回:
        float: 两个框的 IoU 值，取值范围在 [0, 1]。
               如果两个框没有重叠区域，则返回 0.0。
    """
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    inter_x1 = max(x1, a1)
    inter_y1 = max(y1, b1)
    inter_x2 = min(x2, a2)
    inter_y2 = min(y2, b2)
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = (x2-x1)*(y2-y1)
    area2 = (a2-a1)*(b2-b1)
    union = area1 + area2 - inter
    return inter / union

def nms(boxes, scores, thres=0.3):
    """
    执行非极大值抑制（ NMS），从一组候选框中移除重叠度（IoU）过高的框。

    参数:
        boxes (list或ndarray): 候选框数组，形状为 (N, 4)，每个框格式为 (x1, y1, x2, y2)。
        scores (list或ndarray): 每个候选框对应的置信度分数，长度为 N。
        thres (float): IoU 阈值，若两个框的 IoU >= thres 则认为重叠，低分候选框会被抑制。

    返回:
        keep(list): 保留下来的框的索引（相对于输入 `boxes` 的索引），按选取顺序排列。

    说明:
        - 算法流程：先按置信度从高到低排序，每次选取当前最高分的框作为保留，计算它与其它候选框的 IoU，
          将 IoU >= thres 的候选框从候选集中移除，重复直到没有候选框为止。
        - 输入 `boxes` 会被转换为 NumPy 数组以便批量计算。
    """
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1,y1,x2,y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2-x1)* (y2-y1) #每个框的面积
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        # 批量计算当前框 和 其余所有框 的交集坐标
        # np.maximum逐元素取最大值函数
        xx1 = np.maximum(x1[i], x1[order[1:]]) 
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2-xx1)
        h = np.maximum(0, yy2-yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        #  NMS 中用于丢弃与当前框重叠过大的候选框
        order = order[1:][iou < thres]
    return keep

def generate_box(img,topk=2000):
    """
    使用 Felzenszwalb 分割结果为输入图像生成候选目标框。

    参数:
        img: 输入图像，通常为 RGB 图像。
        topk: 返回候选框的最大数量，默认 2000。

    返回:
        list: 去重后的候选框列表，每个候选框格式为 (x1, y1, x2, y2)，
              只保留面积大于等于 200 的框，并返回前 topk 个。
    """

    segments = felzenszwalb(img, scale=200, sigma=0.8, min_size=100)
    proposals = set()

    for label in np.unique(segments):
        mask = (segments == label)
        # 找到满足segments == label像素的位置（ys, xs）
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue
        # 找出这个区域所有像素点的最小和最大 x、y，由此得到一个矩形框
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        if (x2 - x1) * (y2 - y1) < 200:
            continue

        proposals.add((x1, y1, x2, y2))

    return list(proposals)[:topk]

preprocess = transform.Compose([
    transform.ToPILImage(), # 将 NumPy 数组转换为 PIL 图像, transform.Resize需要PIL图像
    transform.Resize((227,227)),#Alexnet输入尺寸
    transform.ToTensor(),
    transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
)

def crop_roi(img, box, device):
    """裁剪并预处理 ROI（Region of Interest）。

    参数:
        img (ndarray): 原始图像，形状为 (H, W, C)，一般为 NumPy 数组（RGB）。
        box (tuple/list): 边界框，格式为 (x1, y1, x2, y2)，表示左上和右下像素坐标（整数）。
        device (torch.device): 返回张量应放置的设备

    返回:
        torch.Tensor 或 None: 如果框无效则返回 None，否则返回形状为 `(1, 3, 227, 227)` 的张量。
    """
    if img is None or len(box) != 4:
        return None

    x1, y1, x2, y2 = map(int, box)
    h, w = img.shape[:2]

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None

    roi = img[y1:y2, x1:x2]
    if roi.size == 0 or roi.ndim != 3:
        return None

    tensor = preprocess(roi).unsqueeze(0).to(device) # (1, 3, 227, 227)
    return tensor


def voc_11point_ap(precisions, recalls):
    """
    VOC2007 11点插值法计算AP（PR 曲线面积近似算法）
    parameters:
        precisions: 按置信降序的精确率列表
        recalls: 对应召回率列表
    return: 
        ap 平均精度
    """
    ap = 0.0
    # 11个召回率采样点
    recall_points = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for thres in recall_points:
        max_p = 0.0
        for p, r in zip(precisions, recalls):
            if r >= thres and p > max_p:
                max_p = p
        ap += max_p
    return ap / 11