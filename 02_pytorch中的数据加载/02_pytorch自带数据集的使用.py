import  torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt

mnist = MNIST(root=r"./datasets", train=True, download=False)
print(mnist)

print(mnist[0][0])

plt.imshow(mnist[0][0], cmap=plt.cm.gray_r)
plt.show()

ret = transforms.ToTensor()(mnist[0][0])    # 将形状(H,W,C)转换为(C,H,W)
print(ret.size())
