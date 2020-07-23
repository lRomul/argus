import argparse
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

import timm

import argus
from argus.callbacks import (
    MonitorCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    LoggingToCSV
)


CIFAR_DATA_DIR = Path('./cifar_data')
EXPERIMENT_DIR = Path('./cifar_simple')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device (default: cuda)')
    return parser.parse_args()


def get_data_loaders(batch_size):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = CIFAR10(root=CIFAR_DATA_DIR, train=True,
                            transform=train_transform, download=True)
    val_dataset = CIFAR10(root=CIFAR_DATA_DIR, train=False,
                          transform=test_transform, download=True)
    train_loader = DataLoader(train_dataset, num_workers=2, drop_last=True,
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, num_workers=2,
                            batch_size=batch_size * 2, shuffle=False)
    return train_loader, val_loader


class CifarModel(argus.Model):
    nn_module = timm.create_model


if __name__ == "__main__":
    args = parse_arguments()
    train_loader, val_loader = get_data_loaders(args.batch_size)

    params = {
        'nn_module': {
            'model_name': 'tf_efficientnet_b0_ns',
            'pretrained': True,
            'num_classes': 10,
            'drop_rate': 0.2,
            'drop_path_rate': 0.2,
        },
        'optimizer': ('AdamW', {'lr': args.lr}),
        'loss': 'CrossEntropyLoss',
        'device': args.device
    }
    model = CifarModel(params)

    callbacks = [
        MonitorCheckpoint(dir_path=EXPERIMENT_DIR, monitor='val_accuracy', max_saves=3),
        EarlyStopping(monitor='val_accuracy', patience=9),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.64, patience=3),
        LoggingToCSV(EXPERIMENT_DIR / 'log.csv')
    ]

    model.fit(train_loader,
              val_loader=val_loader,
              num_epochs=args.epochs,
              metrics=['accuracy'],
              callbacks=callbacks)
