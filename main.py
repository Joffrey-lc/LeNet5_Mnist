# -*- coding: utf-8 -*-
"""
@Time ： 2020/12/17 8:22
@Auth ： LC
@File ：main.py
@IDE ：PyCharm
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Net import mnist_net
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
epochs = 10
batch_size = 64
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader = DataLoader(datasets.MNIST('./data/', train=True, download=True, transform=transforms.Compose([
                           transforms.ToTensor()])),
                          batch_size=batch_size, shuffle=True)
test_loader = DataLoader(datasets.MNIST('./data/', train=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor()])),
                         batch_size=batch_size, shuffle=True)
model = mnist_net().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
model.load_state_dict(torch.load('.\savemodel\mnist_net.pkl'))

for epoch in range(epochs + 1):
    # train
    model.train()
    train_pred = 0
    train_correct = 0
    train_total = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_pred = output.argmax(dim=1, keepdim=True).squeeze(1)
        train_correct += torch.eq(target, train_pred).float().sum().item()
        # print(torch.eq(target, train_pred).float().sum().item())
        train_total += data.shape[0]
        loss.backward()
        optimizer.step()
    # print(train_total, train_correct)
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data_test, target_test in test_loader:
            data_test, target_test = data_test.to(device), target_test.to(device)
            output_test = model(data_test)
            test_loss += criterion(output_test, target_test).item()
            pred = output_test.argmax(dim=1, keepdim=True).squeeze(1)
            test_correct += torch.eq(target_test, pred).float().sum().item()
            test_total += data_test.shape[0]
    # print(test_total)

    test_loss /= test_total

    print("\n第%d个epoch的loss:%f train_acc:%4f test_acc:%4f" % (epoch, loss.item(), train_correct/train_total, test_correct/test_total))

torch.save(model.state_dict(), "./savemodel/mnist_net.pkl")
