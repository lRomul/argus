import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST
import argparse

from argus import Model


def get_data_loaders(train_batch_size, val_batch_size):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(MNIST(download=True, root="./mnist", transform=data_transform, train=True),
                              batch_size=train_batch_size, shuffle=True)

    val_loader = DataLoader(MNIST(download=False, root="./mnist", transform=data_transform, train=False),
                            batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader


class Net(nn.Module):
    def __init__(self, n_classes, p_dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=p_dropout)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, n_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class MnistModel(Model):
    nn_module = Net
    optimizer = torch.optim.SGD
    loss = torch.nn.CrossEntropyLoss


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val_batch_size', type=int, default=64,
                        help='input batch size for validation (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout probability (default: 0.5)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='device (default: cpu)')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    train_loader, val_loader = get_data_loaders(args.train_batch_size, args.val_batch_size)

    params = {
        'nn_module': {'n_classes': 10, 'p_dropout': args.dropout},
        'optimizer': {'lr': args.lr},
        'device': args.device
    }

    model = MnistModel(params)
    print("Result model:", model)
    model.fit(train_loader, val_loader=val_loader, max_epochs=args.epochs)
