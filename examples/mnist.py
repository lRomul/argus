import argparse
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST

from argus import Model, load_model
from argus.callbacks import MonitorCheckpoint, EarlyStopping, \
    ReduceLROnPlateau, LoggingToCSV


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout probability (default: 0.1)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='device (default: cpu)')
    return parser.parse_args()


def get_data_loaders(train_batch_size, val_batch_size):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train_mnist_dataset = MNIST(download=True, root="mnist_data",
                                transform=data_transform, train=True)
    val_mnist_dataset = MNIST(download=False, root="mnist_data",
                              transform=data_transform, train=False)
    train_loader = DataLoader(train_mnist_dataset,
                              batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_mnist_dataset,
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


if __name__ == "__main__":
    args = parse_arguments()
    train_loader, val_loader = get_data_loaders(args.batch_size,
                                                args.batch_size * 2)

    params = {
        'nn_module': {'n_classes': 10, 'p_dropout': args.dropout},
        'optimizer': {'lr': args.lr},
        'device': args.device
    }
    model = MnistModel(params)

    callbacks = [
        MonitorCheckpoint(dir_path='mnist/', monitor='val_accuracy', max_saves=3),
        EarlyStopping(monitor='val_accuracy', patience=9),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3),
        LoggingToCSV('mnist/log.csv')
    ]

    model.fit(train_loader,
              val_loader=val_loader,
              num_epochs=args.epochs,
              metrics=['accuracy'],
              callbacks=callbacks,
              metrics_on_train=True)

    del model
    model_path = Path("mnist/").glob("*.pth")
    model_path = sorted(model_path)[-1]
    print(f"Load model: {model_path}")
    model = load_model(model_path)
    print(model.__dict__)
