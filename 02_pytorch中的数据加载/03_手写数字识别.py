import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 1024

# 1. 准备数据
def get_dataloader(train=True, batch_size=TRAIN_BATCH_SIZE):
    transform_fn = Compose([
        ToTensor(),                                 # 将原图片形状(H, W, C)转换为形状(C, H, W)的tensor
        Normalize(mean=(0.1307,), std=(0.3081,))    # 标准化处理
    ])
    mnist = MNIST(root="./datasets", train=train, transform=transform_fn)
    data_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)
    return data_loader

# 2. 构建模型
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1*28*28, 28)
        self.fc2 = nn.Linear(28, 10)

    def forward(self, x):
        x = x.view(-1, 1*28*28)
        x = self.fc1(x)     # 全连接层
        x = F.relu(x)       # 激活函数
        out = self.fc2(x)   # 输出层
        return out

# 3. 实例化模型、优化器、损失函数
model = Model()
optimizer = Adam(model.parameters(), 0.01)          # Adam优化器
loss_fn = nn.CrossEntropyLoss()                     # 交叉熵损失函数

# 如果之前训练的参数有保存，那么直接加载该参数到模型和优化器中
if os.path.exists("./model/model.pt"):
    model.load_state_dict(torch.load("./model/model.pt"))
    optimizer.load_state_dict(torch.load("./model/optimizer.pt"))

# 4. 循环，进行梯度下降，参数更新
def model_train(epoch):
    model.train(mode=True)  # 模型设置为训练模型
    for i in range(epoch):
        train_data = get_dataloader()   # 加载训练数据
        for (idx, [data, target]) in enumerate(train_data): # 遍历每个Batch
            output = model(data)            # 计算输出
            loss = loss_fn(output, target)  # 前向传播计算损失
            optimizer.zero_grad()           # 梯度置为0
            loss.backward()                 # 反向传播计算参数梯度
            optimizer.step()                # 更新参数
            if idx % 50 == 0:
                print(idx, loss.item())
                # 保存模型和优化器参数
                torch.save(model.state_dict(), "./model/model.pt")
                torch.save(optimizer.state_dict(), "./model/optimizer.pt")

# 5. 评估模型
def model_test():
    loss_list = []
    acc_list = []
    model.eval()    # 设置模型为评估模式
    test_data = get_dataloader(train=False, batch_size=TEST_BATCH_SIZE)
    for (idx, [data, target]) in enumerate(test_data):
        with torch.no_grad():       # 模型评估无需进行梯度计算
            output = model(data)            # 计算模型输出值，output形状为[batch_size, 10]
            # 计算当前batch数据的损失值
            loss_cur = loss_fn(output, target)
            loss_list.append(loss_cur)
            # 计算当前batch数据预测的准确率
            pred = output.max(dim=-1)[-1]   # 取output每行的最大值对应的位置，即预测数字，pred形状为[batch_size]
            # acc_cur = pred.eq(target).float().mean()
            acc_cur = np.mean(pred.numpy() == target.numpy())
            acc_list.append(acc_cur)
    print("平均损失，平均准确率：", np.mean(loss_list), np.mean(acc_list))


if __name__ == '__main__':
    # model_train(3)
    model_test()
