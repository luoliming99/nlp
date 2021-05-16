import torch
from torch.utils.data import Dataset, DataLoader

data_path = r"E:\14. Python Project\08_pytorch nlp实战\02_pytorch中的数据加载\datasets\SMSSpamCollection"

# 完成数据集类
class MyDataSet(Dataset):
    def __init__(self):
        self.lines = open(data_path, encoding="utf_8").readlines()

    def __getitem__(self, index):
        # 获取索引对应位置的一条数据
        cur_line = self.lines[index].strip()    # strip去除回车
        label = cur_line[:4].strip()
        content = cur_line[4:].strip()
        return label, content

    def __len__(self):
        # 返回数据的总数量
        return len(self.lines)

my_dataset = MyDataSet()

data_loader = DataLoader(dataset=my_dataset, batch_size=10, shuffle=True)

if __name__ == "__main__":
    # my_dataset = MyDataSet()
    # for i in range(len(my_dataset)):
    #     print(i, my_dataset[i])

    # for i in data_loader:
    #     print(i)

    # 遍历，获取其中每个batch的结果
    for index, (label, text) in enumerate(data_loader):
        print(index, label, text)
        print("*" * 100)

    print(len(data_loader))