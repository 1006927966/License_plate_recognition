from regonitionmodel import getrecogmodel
from recognition.datadeal import getimg, pre2str
import torch
import os

# 参数路径
dicpath = "/code/wujilong/model/large/netCRNN_10.pth"
# 图像路径
imgdir = "/code/wujilong/data/车牌/车牌测试/licenseplate_test_cut"
# 存储图像结果txt文件
txtpath = "/code/wujilong/data/车牌/车牌测试/licenseplate_test_cut.txt"
# 是否使用gpu
cuda = True

print("[*]! 载入模型！")
model = getrecogmodel()
model.load_state_dict(torch.load(dicpath, map_location="cpu"))
model.eval()

if cuda:
    model.cuda()
print("[*]! 模型载入成功！")

picnames = os.listdir(imgdir)
count = 0
for name in picnames:
    print("推理进度: {}/{}".format(count, len(picnames)))
    if "jpg" not in name and "jpeg" not in name and "png" not in name:
        continue
    picpath = os.path.join(imgdir, name)
    try:
        img = getimg(picpath)
        if cuda:
            img = img.cuda()
        preds = model(img)
        rstr = pre2str(preds)
        print(name)
        print(rstr)
        with open(txtpath, "a") as f:
            f.write("{},{}\n".format(name, rstr))
        count += 1
    except:
        print("[*]! infere Failed: {}".format(name))







