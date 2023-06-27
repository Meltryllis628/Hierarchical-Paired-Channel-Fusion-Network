import cv2
import os
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from model import HPCFNet
from modules.Loss import WeightedCrossEntropyLoss

def train(device, model, dataset, epochs=10, lr=0.01):
    es = 0
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    criterion = WeightedCrossEntropyLoss()
    train_losses = []
    valid_losses = []
    for epoch in range(epochs):
        train_dataset, valid_dataset = random_split(dataset, [90, 10])

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

        model.train()
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        print(f"--- Epoch {epoch+1}/{epochs}: Train loss: {epoch_loss:.4f}", end = "")

        model.eval()
        epoch_loss = 0.0
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            epoch_loss += loss.item()
        epoch_loss /= len(valid_loader)
        valid_losses.append(epoch_loss)
        print(f", valid loss: {epoch_loss:.4f}")
        try:
            os.makedirs(".\\state_dict")
        except:
            pass
        torch.save(model.state_dict(), ".\\state_dict\\{}.pt".format(epoch))
        if epoch >= 10:
            if (epoch_loss - valid_losses[epoch - 2]) > -0.0001:
                es += 1
            else:
                es = 0
            if es >= 5:
                break
    return train_losses, valid_losses