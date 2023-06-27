import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from model import HPCFNet
from modules.Loss import WeightedCrossEntropyLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = HPCFNet(64).to(device)
# a = summary(model, (3, 2048, 244))

x = torch.rand((1, 3, 2048, 224))
y = torch.randint(low=0, high=2, size=(1, 2, 1024, 224), dtype=torch.int32).float()

optimizer = optim.Adam(model.parameters(), lr = 0.001)
criterion = WeightedCrossEntropyLoss()
model.train()
inputs = x.to(device)
labels = y.to(device)
optimizer.zero_grad()
outputs = model(inputs)
_, predicted = torch.max(outputs.data, 1)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
print(loss.item())