Argus
=====

Argus is easy-to-use flexible library for training neural networks in PyTorch.


Warning
=======
The project is in development, so it is not yet suitable for use.


Installation
============

From source:

.. code:: bash

    python setup.py install


Example
=======

.. code-block:: python

    import torch
    from torch import nn
    import torch.nn.functional as F
    from mnist_utils import get_data_loaders

    from argus import Model


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
        train_loader, val_loader = get_data_loaders()

        params = {
            'nn_module': {'n_classes': 10, 'p_dropout': 0.1},
            'optimizer': {'lr': 0.01},
            'device': 'cpu'
        }

        model = MnistModel(params)
        model.fit(train_loader, val_loader=val_loader, max_epochs=10)
