import torch
import matplotlib.pyplot as plt

learning_rate = 0.5

# 1、准备数据
# y = 3x + 0.8
x = torch.rand([500, 1])
y_true = x*3 + 0.8

# 2、通过模型计算y_predict
w = torch.rand([1, 1], requires_grad=True)
b = torch.rand([1, 1], requires_grad=True)
# b = torch.tensor(0, requires_grad=True, dtype=torch.float32)

# 4、通过循环反向传播，更新参数
for i in range(20):

    y_predict = torch.matmul(x, w) + b
    loss = (y_predict - y_true).pow(2).mean()

    if w.grad is not None:
        w.grad.zero_()
    if b.grad is not None:
        b.grad.zero_()

    loss.backward(retain_graph=True)  # 反向传播
    w.data = w.data - learning_rate*w.grad
    b.data = b.data - learning_rate*b.grad

    print("w, b, loss", w.item(), b.item(), loss.item())

plt.figure(figsize=(10, 6))
plt.scatter(x.numpy().reshape(-1), y_true.numpy().reshape(-1), s=5)
y_predict = torch.matmul(x, w) + b
plt.plot(x.numpy().reshape(-1), y_predict.detach().numpy().reshape(-1))  # y_predict中包含grad_fn，因此要先detach
plt.show()