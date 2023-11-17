import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2

# 下载训练集
train_dataset = torchvision.datasets.MNIST(root='./dataset/',
                train=True,
                transform=torchvision.transforms.ToTensor(),
                download=True)

# 下载测试集
test_dataset = torchvision.datasets.MNIST(root='./dataset/',
               train=False,
               transform=torchvision.transforms.ToTensor(),
               download=True)

batch_size = 9

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                      batch_size=batch_size,
                      shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                     batch_size=batch_size,
                     shuffle=True)

pass