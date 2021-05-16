import torch
from torch import nn
from torch.optim import SGD
import matplotlib.pyplot as plt

# 定义一个device对象
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 0、准备数据
x = torch.rand([500, 1]).to(device) # 这个参数要传到model中，也应当和model的device一样
y_true = 3*x + 0.8

# 1、定义模型
class MyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

# 2、实例化模型，优化器类实例化，loss实例化
my_linear = MyLinear().to(device)   # 模型to device，由于模型内部参数不是我们手动定义，所以参数会自动to device
optimizer = SGD(my_linear.parameters(), 0.01)    # learning_rate = 0.5
loss_fn = nn.MSELoss()

# 3、循环，进行梯度下降，参数更新
for i in range(500):
    # 得到预测值
    y_predict = my_linear(x)
    loss = loss_fn(y_predict, y_true)
    # 梯度置为0
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 参数的更新
    optimizer.step()
    if i % 10 == 0:
        params = list(my_linear.parameters())
        print(loss.item(), params[0].item(), params[1].item())

# 4、模型评估
my_linear.eval()
y_predict = my_linear(x)
plt.scatter(x.data.numpy(), y_true.data.numpy(), s=5)
plt.plot(x.data.numpy(), y_predict.data.numpy())
plt.show()



