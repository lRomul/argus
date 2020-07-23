import os
import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import SyncBatchNorm
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms
from torchvision.datasets import CIFAR10

import timm

import argus
from argus.callbacks import (
    MonitorCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    LoggingToCSV,
    LoggingToFile
)

torch.backends.cudnn.benchmark = True

CIFAR_DATA_DIR = Path('./cifar_data')
EXPERIMENT_DIR = Path('./cifar_advanced')


def get_linear_scaled_lr(base_lr, batch_size, base_batch_size=128):
    return base_lr * (batch_size / base_batch_size)


def get_data_loaders(batch_size, distributed, local_rank):
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

    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_dataset,
                                           num_replicas=dist.get_world_size(),
                                           rank=local_rank,
                                           shuffle=True)

    train_loader = DataLoader(train_dataset, num_workers=2, drop_last=True,
                              batch_size=batch_size, sampler=train_sampler,
                              shuffle=train_sampler is None)
    val_loader = DataLoader(val_dataset, num_workers=2,
                            batch_size=batch_size * 2, shuffle=False)
    return train_loader, val_loader


class CifarModel(argus.Model):
    nn_module = timm.create_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')

    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')

    if args.distributed:
        args.world_batch_size = args.batch_size * dist.get_world_size()
    else:
        args.world_batch_size = args.batch_size
    print("World batch size:", args.world_batch_size)

    train_loader, val_loader = get_data_loaders(args.batch_size,
                                                args.distributed,
                                                args.local_rank)

    params = {
        'nn_module': {
            'model_name': 'tf_efficientnet_b0_ns',
            'pretrained': True,
            'num_classes': 10,
            'drop_rate': 0.2,
            'drop_path_rate': 0.2,
        },
        'optimizer': ('AdamW', {
            'lr': get_linear_scaled_lr(args.lr, args.world_batch_size)
        }),
        'loss': 'CrossEntropyLoss',
        'device': 'cuda'
    }

    model = CifarModel(params)

    if args.distributed:
        model.nn_module = SyncBatchNorm.convert_sync_batchnorm(model.nn_module)
        model.nn_module = DistributedDataParallel(model.nn_module.to(args.local_rank),
                                                  device_ids=[args.local_rank],
                                                  output_device=args.local_rank)
        if args.local_rank:
            model.logger.disabled = True
    else:
        model.set_device('cuda')

    callbacks = []
    if args.local_rank == 0:
        callbacks += [
            MonitorCheckpoint(dir_path=EXPERIMENT_DIR, monitor='val_accuracy', max_saves=3),
            LoggingToCSV(EXPERIMENT_DIR / 'log.csv'),
            LoggingToFile(EXPERIMENT_DIR / 'log.txt')
        ]

    callbacks += [
        EarlyStopping(monitor='val_accuracy', patience=9),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.64, patience=3),
    ]

    if args.distributed:
        @argus.callbacks.on_epoch_complete
        def schedule_sampler(state):
            state.data_loader.sampler.set_epoch(state.epoch + 1)
        callbacks += [schedule_sampler]

    model.fit(train_loader,
              val_loader=val_loader,
              num_epochs=args.epochs,
              metrics=['accuracy'],
              callbacks=callbacks)
