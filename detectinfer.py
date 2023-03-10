import os
import numpy as np
from yolov6.utils.datadeal import resize, plot_one_box, getmaxbox, getmaxscorebox
from yolov6.utils.nms import non_max_suppression
import cv2
from detectmodel import detectModel, getmodel
# 检测模型参数路径
detectdictpath = "/code/wujilong/code/YOLOv6/runs/train/licence/weights/last_ckpt.pt"
#画框图像存储路径
detectsavedir = "/code/wujilong/data/车牌/车牌测试/licenseplate_test_plot"
#剪切后的车牌存储路径
cutsavedir = "/code/wujilong/data/车牌/车牌测试/licenseplate_test_cut"
# 推理图像路径
imgdir = "/code/wujilong/data/车牌/车牌测试/licenseplate_test"
#是否使用GPU
cuda = True
#是否需要画框
plot = True
#是否在原图中剪切出车牌
cut = True

# 加载模型参数
print("[*]! model is loading now!")
#model = detectModel(detectdictpath)
model = getmodel(detectdictpath)
if cuda:
    model.cuda()
model.eval()
print("[*]! model loads finished!")

#开始车牌检测阶段推理
badpic = 0 #统计没有召回数量
count = 0 #跟踪车牌推理进度
picnames = os.listdir(imgdir)
for name in picnames:
    if "jpg" not in name and "jpeg" not in name and "png" not in name:
        continue
    try:
        picpath = os.path.join(imgdir, name)
        img = resize(picpath)
        if cuda:
            img = img.cuda()
    except:
        print("[*]! reading Failed: {}".format(name))
        continue
    print("[*]! infering : {}/{}".format(count, len(picnames)))
    try:
        pre = model(img)
        outputs = non_max_suppression(pre, 0.4, 0.5)
        if outputs[0].shape[0] == 0:
            badpic += 1
            continue
        else:
            oimg = cv2.imread(picpath)
            outputs = outputs[0].detach().cpu().numpy()
            x1, y1, x2, y2 = getmaxscorebox(outputs, oimg)
            print(x1, y1, x2, y2)
            print(outputs)
            if cut == True:
                os.makedirs(cutsavedir, exist_ok=True)
                savecut = os.path.join(cutsavedir, name)
                subimg = oimg[y1:y2, x1:x2,:]
                cv2.imwrite(savecut, subimg)

            if plot==True:
                os.makedirs(detectsavedir, exist_ok=True)
                plot_one_box([x1, y1, x2, y2], oimg)
                saveplot = os.path.join(detectsavedir, name)
                cv2.imwrite(saveplot, oimg)
        count += 1
    except:
        print("[*]! infere Failed: {}".format(name))
        continue
print("[*]! no recall num is: {}".format(badpic))









