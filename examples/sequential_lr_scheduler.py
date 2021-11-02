from torchvision import transforms
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, ConstantLR, ExponentialLR

import argus
from argus.callbacks.lr_schedulers import LRScheduler


def get_data_loaders(batch_size):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_dataset = CIFAR10(root='./cifar_data', train=True,
                            transform=data_transform, download=True)
    val_dataset = CIFAR10(root='./cifar_data', train=False,
                          transform=data_transform, download=True)
    train_loader = DataLoader(train_dataset, num_workers=2, drop_last=True,
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, num_workers=2,
                            batch_size=batch_size * 2, shuffle=False)
    return train_loader, val_loader


class ResNetModel(argus.Model):
    nn_module = resnet18


if __name__ == "__main__":

    def get_lr_scheduler(optimizer):
        scheduler1 = ConstantLR(optimizer, factor=0.1, total_iters=2)
        scheduler2 = ExponentialLR(optimizer, gamma=0.9)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[2]
        )
        return scheduler

    lr_scheduler = LRScheduler(get_lr_scheduler)

    train_loader, val_loader = get_data_loaders(batch_size=256)
    params = {
        'nn_module': {
            'pretrained': False,
        },
        'optimizer': ('Adam', {'lr': 0.01}),
        'loss': 'CrossEntropyLoss',
        'device': 'cuda'
    }

    model = ResNetModel(params)

    model.fit(train_loader,
              val_loader=val_loader,
              num_epochs=10,
              metrics=['accuracy'],
              callbacks=[lr_scheduler])
