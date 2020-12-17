# -*- coding: utf-8 -*-
"""
@Time ： 2020/12/17 9:20
@Auth ： LC
@File ：application.py
@IDE ：PyCharm
"""
import torch
from Net import mnist_net
from PIL import Image
import numpy as np

model = mnist_net()
model.load_state_dict(torch.load('.\savemodel\mnist_net.pkl'))
num_pre = input("输入要识别的个数（决定程序循环几次）:")
for i in range(int(num_pre)):
    path = input('输入图片的地址：')
    image = Image.open(path)
    image = image.resize((28, 28), Image.ANTIALIAS)

    image_arr = np.array(image.convert('L'))  # 转为numpy 且是灰度图

    image_arr = (255 - image_arr)/255.  # 反转颜色和归一化
    data = torch.from_numpy(image_arr).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor)
    pre = model(data)
    print("预测结果为:", pre.argmax(dim=1).item())
