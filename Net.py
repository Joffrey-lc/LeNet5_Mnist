# -*- coding: utf-8 -*-
"""
@Time ： 2020/12/17 8:22
@Auth ： LC
@File ：Net.py
@IDE ：PyCharm
"""
import torch.nn as nn
import torch


class mnist_net(nn.Module):
    def __init__(self):
        super(mnist_net, self).__init__()
        self.lenet5_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1)),
        )
        self.lenet5_dense = nn.Sequential(
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.lenet5_conv(x)
        x = x.view(-1, 120)
        x = self.lenet5_dense(x)
        return x


if __name__ == '__main__':
    a = torch.rand([10, 1, 28, 28])
    print(a.shape)
    net = mnist_net()
    out = net(a)
    print(out.shape)
