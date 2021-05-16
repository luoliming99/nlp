import torch
from torch import nn
from torch.optim import SGD
import matplotlib.pyplot as plt

# 0. 准备数据
x = torch.rand([500,1])
y_ture = 3*x + 0.8

# 1. 定义模型
class MyLinear(nn.Module):
    def __init__(self):
        # 继承父类的init
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

# 2. 实例化模型，优化器，损失函数
my_linear = MyLinear()
optimizer = SGD(my_linear.parameters(), 0.01)
loss_fn = nn.MSELoss()

# 3. 循环，进行梯度下降，参数更新
for i in range(2000):
    # 计算预测值
    y_predict = my_linear(x)
    # 计算损失
    loss = loss_fn(y_predict, y_ture)
    # 梯度清零
    optimizer.zero_grad()
    # 反向传播计算梯度
    loss.backward()
    # 参数更新
    optimizer.step()

    # 打印
    if i % 100 == 0:
        paras = list(my_linear.parameters())
        print(loss.item(), paras[0].item(), paras[1].item())

# 4. 模型评估
my_linear.eval()    # 将my_linear.train赋值为False，进入评估模式
y_predict = my_linear(x)
plt.figure(figsize=(10, 6))
plt.scatter(x.numpy(), y_ture.numpy(), s=5)
plt.plot(x.numpy(), y_predict.data.numpy())
plt.show()