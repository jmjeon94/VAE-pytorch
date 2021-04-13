import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
from model import VAE

# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

# Data Loader (Input Pipeline)
batch_size = 256
epochs = 50
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = VAE()
model.to(device)

optimizer = optim.Adam(model.parameters())

def criterion(x, recon_x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

for epoch in range(epochs):

    for x_train, _ in train_loader:
        losses = 0
        x_train = x_train.to(device)

        optimizer.zero_grad()

        pred, mu, log_var = model(x_train)

        loss = criterion(x_train, pred, mu, log_var)
        loss.backward()

        optimizer.step()

        losses += loss.item()

    print(f'Epoch:{epoch+1:02}, Loss:{losses/len(train_loader):.3f}')
    torch.save(model.state_dict(), f'./checkpoints/epoch_{epoch+1:03}.pth')

