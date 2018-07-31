import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import argparse

import argus
from argus import Model
from argus.engine import State
from argus.callbacks import Checkpoint


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--val_batch_size', type=int, default=128,
                        help='input batch size for validation (default: 128)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='device (default: cpu)')
    return parser.parse_args()


class MnistVaeDataset(Dataset):
    def __init__(self, mnist_dataset):
        super().__init__()
        self.dataset = mnist_dataset

    def __getitem__(self, index):
        input, target = self.dataset[index]
        return input, input

    def __len__(self):
        return len(self.dataset)


def get_data_loaders(train_batch_size, val_batch_size):
    train_mnist_dataset = MNIST(download=True, root="mnist_data",
                                transform=ToTensor(), train=True)
    val_mnist_dataset = MNIST(download=False, root="mnist_data",
                              transform=ToTensor(), train=False)
    train_loader = DataLoader(MnistVaeDataset(train_mnist_dataset),
                              batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(MnistVaeDataset(val_mnist_dataset),
                            batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader


class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
class VaeLoss(nn.modules.Module):
    def forward(self, inp, trg):
        recon, mu, logvar = inp
        bce = F.binary_cross_entropy(recon, trg.view(-1, 784), size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce + kld


class VaeMnistModel(Model):
    nn_module = VAE
    loss = VaeLoss


@argus.callbacks.on_epoch_complete
def save_reconstructions(state: State):
    data = state.step_output['target']
    recon, _, _ = state.step_output['prediction']
    batch_size = data.size(0)
    num_images = min(batch_size, 8)
    comparison = torch.cat([data[:num_images],
                           recon.view(batch_size, 1, 28, 28)[:num_images]])
    save_image(comparison.to('cpu'),
               f'mnist_vae/recon_epoch_{state.epoch}.png', nrow=num_images)


if __name__ == "__main__":
    args = parse_arguments()
    train_loader, val_loader = get_data_loaders(args.train_batch_size, args.val_batch_size)

    params = {
        'optimizer': ('Adam', {'lr': args.lr}),
        'device': args.device
    }
    model = VaeMnistModel(params)

    callbacks = [
        Checkpoint(dir_path='mnist_vae/checkpoints', period=3)
    ]

    model.fit(train_loader,
              val_loader=val_loader,
              max_epochs=args.epochs,
              callbacks=callbacks,
              val_callbacks=[save_reconstructions])
