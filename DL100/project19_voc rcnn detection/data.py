from PIL import Image, ImageDraw
from torchvision.datasets import VOCDetection
from utils import show_voc_image_with_boxes, generate_box,compute_iou

train_data = VOCDetection(root='./data/VOC2012/', year='2012', image_set='train', download=False)
test_data = VOCDetection(root='./data/VOC2012/', year='2007', image_set='test', download=False)
'''
print("数据集大小：", len(test_data))
img, target = test_data[0]

print("图片类型：", type(img))
print("图片尺寸：", img.size)
print("第一条标注：", target)
img.show()  # 显示第一张图片

print("数据集大小：", len(train_data))
img, target = train_data[0]

print("图片类型：", type(img))
print("图片尺寸：", img.size)
print("第一条标注：", target)
img.show()  # 显示第一张图片

show_voc_image_with_boxes(train_data, 0)
'''


'''
# 把候选框画到图片上
img_draw = img.copy()
draw = ImageDraw.Draw(img_draw)
for i, (x1, y1, x2, y2) in enumerate(proposals):
    draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
    draw.text((x1, max(0, y1 - 10)), str(i), fill='red')

img_draw.show()
print('候选框数量：', len(proposals))
'''

# 解析VOC xml获取真实框
def parse_voc_target(target_dict):
    '''
    解析VOC数据集单张图片的标注字典，提取所有目标类别与边框真值
    Args:
        target_dict (dict): VOC xml解析后转换得到的顶层标注字典,经过Dataloader后的target_dict
    Returns:
        list[dict]: 真值列表，每个元素为单个目标字典
            - "cls" (str): 目标类别名称
            - "box" (list[int]): 像素坐标边框 [xmin, ymin, xmax, ymax]
    '''
    gts = []
    anno = target_dict["annotation"]
    for obj in anno["object"]:
        cls = obj["name"][0] # obj["name"]是list，ex:['dog']
        bbox = obj["bndbox"]
        x1 = int(bbox["xmin"][0])
        y1 = int(bbox["ymin"][0])
        x2 = int(bbox["xmax"][0])
        y2 = int(bbox["ymax"][0])
        gts.append({"cls": cls, "box": [x1, y1, x2, y2]})
    return gts


# 单张图提取正负样本（RCNN标准规则）
def get_pos_neg_samples(img, gts):
    '''
    Args:
        img (ndarray): 输入图像
        gts (list[dict]): 当前图片所有真实标注框，每个字典键：
            "cls": 类别名称字符串；"box": GT坐标(x1, y1, x2, y2)

    Returns:
        pos_samples (list[tuple]): 全部正样本ROI坐标（IoU≥0.5候选框 + 额外补充的GT框）
        pos_cls_ids (list[str]): 每个正样本对应的物体类别名，与pos_samples一一对应
        pos_gt_boxes (list[tuple]): 每个正样本匹配的真实GT框坐标，用于后续边框回归训练
        neg_samples (list[tuple]): 均衡裁剪后的负样本候选框，仅IoU≤0.1的背景区域
    '''
    proposals = generate_box(img)
    pos_samples = []
    pos_cls_ids = []
    pos_gt_boxes = []
    neg_samples = []

    print("候选框数量 len(proposals):", len(proposals))
    print("真值框数量 len(gts):", len(gts))
    if len(proposals) > 0:
        print("第一个候选框坐标:", proposals[0])
    if len(gts) > 0:
        print("第一个GT框坐标:", gts[0]["box"])

    for prop in proposals:
        max_iou = 0.0
        match_gt = None
        for gt in gts:
            iou = compute_iou(prop, gt["box"])
            if iou > max_iou:
                max_iou = iou
                match_gt = gt

        # print("单个proposal最大IoU:", max_iou)
        # 正样本 IoU >=0.5
        if max_iou >= 0.5:
            pos_samples.append(prop)
            pos_cls_ids.append(match_gt["cls"])
            pos_gt_boxes.append(match_gt["box"])
        # 负样本 IoU <=0.1
        elif max_iou <= 0.1:
            neg_samples.append(prop)
    # 将ground Truth标注框加入每个类的正样本，确保正样本数量不会为0
    for gt in gts:
        pos_samples.append(gt["box"])
        pos_cls_ids.append(gt["cls"])
        pos_gt_boxes.append(gt["box"])
    # 平衡正负，负样本最多正样本3倍
    neg_samples = neg_samples[:len(pos_samples)*3]
    return pos_samples, pos_cls_ids, pos_gt_boxes, neg_samples

if __name__ == '__main__' :
    img, target = train_data[0]
    print(target)
    proposals = generate_box(img)
    gts = parse_voc_target(target)
    pos_samples, pos_cls_ids, pos_gt_boxes, neg_samples = get_pos_neg_samples(proposals,gts)
    print(pos_samples)
    print(pos_cls_ids)
    print(pos_gt_boxes)
    print(neg_samples)

    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    for i, (x1, y1, x2, y2) in enumerate(proposals):
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        draw.text((x1, max(0, y1 - 10)), str(i), fill='red')

    img_draw.show()
    print('候选框数量：', len(proposals))