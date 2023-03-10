import cv2
import torch
import numpy as np
import random


def resize(path, msize=640):
    oimg = cv2.imread(path)
    img = cv2.resize(oimg, (msize, msize))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img

def plot_one_box(x, img, color=None, label=None, line_thickness=None):#x（xy,xy） line_thickness=3, label = '%s %.2f' % (names[int(cls)], conf)
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))# nms之后xyxy期中为左上角点，左下角点
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

#如果有多个框那么选择面积最大的那一个, 将坐标放缩成原始图像坐标。可用于剪切以及画框
#

def getmaxbox(boxes, oimg, msize=640):
    boxes[boxes<0] = 0 #纺织输出
    if boxes.shape[0] == 1:
        maxbox = boxes[0]
    else:
        areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
        sorts = np.argsort(-areas) # 面积从大到小
        maxbox = boxes[sorts[0]]
    x1, y1, x2, y2 = maxbox[0], maxbox[1], maxbox[2], maxbox[3]
    h,w = oimg.shape[:2]
    hratio = h/640
    wratio = w/640
    x1 = int(x1*wratio)
    x2 = int(x2*wratio)
    y1 = int(y1*hratio)
    y2 = int(y2*hratio)
    return x1, y1, x2, y2

def getmaxscorebox(boxes, oimg, msize=640):
    boxes[boxes < 0] = 0  # 纺织输出
    if boxes.shape[0] == 1:
        maxbox = boxes[0]
    else:
        scores = boxes[:,4]
        sorts = np.argsort(-scores)  # 分数从大到小
        maxbox = boxes[sorts[0]]
    x1, y1, x2, y2 = maxbox[0], maxbox[1], maxbox[2], maxbox[3]
    h, w = oimg.shape[:2]
    hratio = h / 640
    wratio = w / 640
    x1 = int(x1 * wratio)
    x2 = int(x2 * wratio)
    y1 = int(y1 * hratio)
    y2 = int(y2 * hratio)
    return x1, y1, x2, y2


