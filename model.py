import torch.nn as nn
import torch
import numpy as np

class model(nn.Module):
    def __init__(self, layer_num=4):
        super(model, self).__init__()
        self.layer_num = layer_num
        # self.w1 = torch.from_numpy(0.5 * np.ones((6,  15),dtype=np.int16))
        # self.w2 = torch.from_numpy(0.5 * np.ones((15, 20),dtype=np.int16))
        # self.w3 = torch.from_numpy(0.5 * np.ones((20, 15),dtype=np.int16))
        # self.w4 = torch.from_numpy(0.5 * np.ones((15, 10),dtype=np.int16))
        # print(self.w1)
        self.w1 = nn.Linear(6,15)
        self.w2 = nn.Linear(15,20)
        self.w3 = nn.Linear(20,15)
        self.w4 = nn.Linear(15,10)


    def forward(self, x):
        # x = x * self.w1
        # x = x * self.w2
        # x = x * self.w3
        # y = x * self.w4
        x = self.w1(x)
        x = self.w2(x)
        x = self.w3(x)
        y = self.w4(x)
        return y


