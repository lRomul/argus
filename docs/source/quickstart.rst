Quick start
===========

`Link to quick start jupyter notebook. <https://github.com/lRomul/argus/blob/master/examples/quickstart.ipynb>`_

Simple example
--------------

Define a PyTorch model.

.. code-block:: python

    import torch
    from torch import nn
    import torch.nn.functional as F

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


Define a :class:`~argus.model.Model` with ``nn_module``, ``optimizer``, ``loss`` attributes. Each value must be a class
or function that returns object (``torch.nn.Module`` for loss and nn_module, ``torch.optim.Optimizer`` for optimizer).

.. code-block:: python

    from argus import Model

    class MnistModel(Model):
        nn_module = Net
        optimizer = torch.optim.SGD
        loss = torch.nn.CrossEntropyLoss


Create instance of ``MnistModel`` with specific parameters. Net will be initialized like
``Net(n_classes=10, p_dropout=0.1)``. Same logic for optimizer ``torch.optim.SGD(lr=0.01)``. Loss will be created
without arguments ``torch.nn.CrossEntropyLoss()``.

.. code-block:: python

    params = {
        'nn_module': {'n_classes': 10, 'p_dropout': 0.1},
        'optimizer': {'lr': 0.01},
        'device': 'cpu'
    }

    model = MnistModel(params)


Download MNIST dataset. Create validation and training PyTorch data loaders.

.. code-block:: python

    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, ToTensor, Normalize
    from torchvision.datasets import MNIST

    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train_mnist_dataset = MNIST(download=True, root="mnist_data",
                                transform=data_transform, train=True)
    val_mnist_dataset = MNIST(download=False, root="mnist_data",
                              transform=data_transform, train=False)
    train_loader = DataLoader(train_mnist_dataset,
                              batch_size=64, shuffle=True)
    val_loader = DataLoader(val_mnist_dataset,
                            batch_size=128, shuffle=False)


Use callbacks and start train a model for 50 epochs.

.. code-block:: python

    from argus.callbacks import MonitorCheckpoint, EarlyStopping, ReduceLROnPlateau

    callbacks = [
        MonitorCheckpoint(dir_path='mnist', monitor='val_accuracy', max_saves=3),
        EarlyStopping(monitor='val_accuracy', patience=9),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3)
    ]

    model.fit(train_loader,
              val_loader=val_loader,
              max_epochs=50,
              metrics=['accuracy'],
              callbacks=callbacks)


More flexibility
----------------

Argus can help you simplify the experiments with different architectures, losses, and optimizers. Let's define a
:class:`~argus.model.Model` with two models via a dictionary. If you want to use PyTorch losses and optimizers it's not
necessary to define them in argus model.

.. code-block:: python

    from torchvision.models import resnet18

    class FlexModel(Model):
        nn_module = {
            'net': Net,
            'resnet18': resnet18
        }


Create a model instance. Parameters for nn_module is a tuple where the first element is a name, second is arguments.
PyTorch losses and optimizers can be selected by a string with a class name.

.. code-block:: python

    params = {
        'nn_module': ('resnet18', {
            'pretrained': False,
            'num_classes': 1
        }),
        'optimizer': ('Adam', {'lr': 0.01}),
        'loss': 'CrossEntropyLoss',
        'device': 'cuda'
    }

    model = FlexModel(params)


Argus allows managing different combinations of your pipeline.

If you need for more flexibility you can:

* Override methods of :class:`~argus.model.Model`. For example :meth:`~argus.model.Model.train_step` and :meth:`~argus.model.Model.val_step`.
* Create custom :class:`~argus.callbacks.Callback`.
* Use custom :class:`~argus.metrics.Metric`.

