# License_plate_recognition
   工程中包含检测和识别两个部分，由于车牌训练数据集合为CCPD，所以识别蓝牌比较准，绿牌可能会差一点。
   工程使用YOLOV6+CRNN模型进行车牌识别。
   车牌识别推理步骤
   （1）使用文件detectinfer.py对车牌进行检测。可以获取车牌的剪切图像，以及车牌检测画框图像。
   （2）使用文件recoginfer.py对车牌进行识别。可以获取车牌的预测结果。
   模型提取如下
   链接: https://pan.baidu.com/s/1Jl0-8oqX05OuXK6aKoot2g  密码: 5o5u
   （1）其中last_ckpt.pt是检测模型，将下载后的路径放到文件detectinfer.py中
   （2）netCrnn_10.pth是识别模型，将下载后的路径放到recoginfer.py中
