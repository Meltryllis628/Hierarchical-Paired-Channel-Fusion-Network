import cv2
import os
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from model import HPCFNet
from data.processing import make_dataset
from train import train
import numpy as np
from modules.Loss import WeightedCrossEntropyLoss

model = HPCFNet(64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
make_dataset()
dataset = torch.load("dataset.pt")
model.to(device)

def view_output(i, criterion):
    with torch.no_grad():
        img = dataset[i][0].to(device)
        img = img.unsqueeze(0)
        file_mask = ".\\TSUNAMI\\mask\\{:08d}.png".format(i)
        output = model(img)
        y_1 = np.array(cv2.imread(file_mask, cv2.IMREAD_GRAYSCALE)).T // 255
        y_0 = np.ones_like(y_1) - y_1
        label = torch.tensor(np.stack([y_0, y_1])).float().unsqueeze(0).to(device)
        loss = criterion(output, label)
        _, predicted = torch.max(output.data, 1)
        b = predicted.int().cpu().numpy().transpose(2, 1, 0)*255
        r = np.expand_dims(np.array(cv2.imread(file_mask, cv2.IMREAD_GRAYSCALE)), axis=2)
        g = np.zeros_like(b)
        out = np.concatenate((b,g,r), axis = 2)
        cv2.imwrite(".\\outputs\\{}-{}.jpg".format(i, loss), out)
        return loss.item()


train_losses = []
valid_losses = []
train_losses, valid_losses = train(device, model, dataset, 100, 0.001)

try:
    os.makedirs(".\\outputs")
except:
    pass
loss = 0
model.to(device)
model.eval()
for i in range(100):
    loss += view_output(i, nn.CrossEntropyLoss())
print(loss/100)