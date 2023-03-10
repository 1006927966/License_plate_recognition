from PIL import Image
from torchvision import transforms
import torch
from torch.autograd import Variable

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ] # 最后一个是空白

def getimg(picpath, imgsize=(100, 32)):
    image = Image.open(picpath).convert('L')
    image = image.resize(imgsize, Image.BILINEAR)
    image = transforms.ToTensor()(image)
    image.sub_(0.5).div_(0.5)
    image = image.unsqueeze(0)
    image = Variable(image)
    return image


def getnorepeate(pres):
    norepeate = []
    pre_c = pres[0]
    if pre_c != len(CHARS) -1:
        norepeate.append(pre_c)
    for pre in pres:
        if(pre_c == pre) or (pre == len(CHARS)-1): #不是空白或者重复
            if pre==len(CHARS) -1:#如果是空白则前置是空白
                pre_c = pre
            continue
        norepeate.append(pre)
        pre_c = pre
    return norepeate


def pre2str(preds):
    preds = preds.log_softmax(2)
    _, preds = preds.max(2)  # 结果预测
    preds = preds.transpose(1, 0)  # 每一个图像的预测结果B T
    preds = preds.data.detach().cpu().numpy()[0]
    norepeate = getnorepeate(preds)
    if len(norepeate) == 0:
        return "badpred"
    strs = [CHARS[index] for index in norepeate]
    return "".join(strs)


