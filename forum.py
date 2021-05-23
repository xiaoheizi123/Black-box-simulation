import random
import numpy as np
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import model
import torch
from torch.autograd import Variable

# 模拟一个函数, 得到输入 输出变量
# y1, y2, y3, y4, y5, y6, y7, y8, y9, y10 = 3A^4 + 5B^3 + 7C^2 + 2D^5 + 9E^3 + F^2

def cal_y_from_x(x_list, factor_list, power_list):
    # x_list: 1x6
    # factor_list: [3, 5, 7, 2,9,1]
    # power_list: [4,3,2,5,3,2]
    # target: 得到模拟数据以训练
    factor_list_len = len(factor_list)
    inzi = []
    for i in range(factor_list_len):
        aa =factor_list[i]*pow(x_list[i], power_list[i])/10000
        inzi.append(aa)
    # index 0~5
    yy = [-inzi[0] + inzi[3] + inzi[2], inzi[0] -inzi[1], inzi[1] + inzi[2]- inzi[5], inzi[3], inzi[4],
          inzi[3] - inzi[5], inzi[1] + inzi[4], inzi[2]- inzi[5],inzi[5], inzi[1]+inzi[3]]
    return yy

def prepare_x_y(train_num):
    xx, yy = [], []
    factor_list = [3, 5, 7, 2, 9, 1]
    power_list = [4,3,2,5,3,2]
    for j in range(train_num):
        x = [random.uniform(0, 10) for i in range(6)]
        y = cal_y_from_x(x, factor_list, power_list)
        xx.append(x)
        yy.append(y)
    return xx, yy


def train(Model, X, Y,optimizer):
    optimizer.zero_grad()
    cal = 10
    loss = 0
    n = 0
    for j in range(len(X)//cal):
        X_now = Variable(torch.from_numpy((np.array(X[j:(j+1)])).astype(np.float32)),requires_grad=True)
        y = Model(X_now)
        # print(y)
        Y_np = torch.from_numpy(np.array(Y[j:(j+1)]).astype(np.float32))
        loss_temp = sum([abs(y[ii] - Y_np[ii]) for ii in range(len(y))])
        loss += loss_temp
        n += 1
    # print('loss is:', loss)
    loss.backward(loss)
    optimizer.step()


if __name__ == "__main__":
    train_sample_num = 100
    train_epoches = 250
    factor_num = 5
    out_num = 10
    layer_num = 4
    Model = model.model(layer_num)
    print(Model)
    optimizer = optim.SGD(Model.parameters(), lr=0.00001, momentum=0.9, nesterov=True)
    for i in range(train_epoches):
        X , Y = prepare_x_y(train_sample_num)
        train(Model, X, Y, optimizer)
        xx, yy = prepare_x_y(train_sample_num//2)
        xx_test = Variable(torch.from_numpy((np.array(xx)).astype(np.float32)))[:1]
        yy_test = torch.from_numpy((np.array(yy)).astype(np.float32)).detach()[:1]
        YY = Model(xx_test)
        differ_dis = sum([abs(YY[jj]- yy_test[jj]) for jj in range(len(YY))])
        # print('YY is:     ', YY.detach().numpy().tolist())
        # print('yy_test is:', yy_test.numpy().tolist())
        if(sum(differ_dis.detach())/len(YY))<1:
            print('to save')
            torch.save(Model, str(i)+".pt")




