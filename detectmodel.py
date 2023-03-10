import torch
from yolov6.models.yolo import build_model
from yolov6.utils.configs import Config

#yolov6模型配置文件
def detectModel(dicpath):
    yamlpath = "./configs/yolov6_tiny.py"
    cfg = Config.fromfile(yamlpath)
#模型训练参数路径,选定自己的路径
    statedict = torch.load(dicpath, map_location="cpu")
    detectmodel = build_model(cfg, 2, "cpu")
    detectmodel.state_dict(statedict)
    return detectmodel


def getmodel(modelpath):
    model = torch.load(modelpath, map_location='cpu')['model']
    return model.float()



