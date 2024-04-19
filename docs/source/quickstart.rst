Quick start
===========

`Link to quick start jupyter notebook. <https://github.com/lRomul/argus/blob/master/examples/quickstart.ipynb>`_

Simple example
--------------

1. Define a PyTorch model.

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


2. Define a :class:`argus.model.Model` with ``nn_module``, ``optimizer``, ``loss`` attributes. Each value must be a class
or function that returns object (:class:`torch.nn.Module` for loss and nn_module, :class:`torch.optim.Optimizer` for optimizer).

.. code-block:: python

    import argus

    class MnistModel(argus.Model):
        nn_module = Net
        optimizer = torch.optim.SGD
        loss = torch.nn.CrossEntropyLoss


3. Create an instance of ``MnistModel`` with the specified parameters. Net will be initialized like
``Net(n_classes=10, p_dropout=0.1)``. The same logic is applied for the optimizer ``torch.optim.SGD(lr=0.01)``.
Loss will be created without any arguments ``torch.nn.CrossEntropyLoss()``. The model will use the CPU.

.. code-block:: python

    params = {
        'nn_module': {'n_classes': 10, 'p_dropout': 0.1},
        'optimizer': {'lr': 0.01},
        'device': 'cpu'
    }

    model = MnistModel(params)


4. Download MNIST dataset. Create validation and training PyTorch data loaders.

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


5. Define some callbacks and start training the model for 50 epochs.
As metrics, you can use instances of classes inherit from :class:`argus.metrics.Metric` or they names :attr:`argus.metrics.Metric.name`.

.. code-block:: python

    from argus.callbacks import MonitorCheckpoint, EarlyStopping, ReduceLROnPlateau

    callbacks = [
        MonitorCheckpoint(dir_path='mnist', monitor='val_accuracy', max_saves=3),
        EarlyStopping(monitor='val_accuracy', patience=9),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3)
    ]

    model.fit(train_loader,
              val_loader=val_loader,
              num_epochs=50,
              metrics=['accuracy'],  # or [argus.metrics.CategoricalAccuracy()]
              callbacks=callbacks)

6. Load the model from the best checkpoint.

.. code-block:: python

    from pathlib import Path

    model_path = Path("mnist/").glob("*.pth")
    model_path = sorted(model_path)[-1]
    print(f"Load model: {model_path}")
    loaded_model = argus.load_model(model_path)
    print(loaded_model)

More flexibility
----------------

Argus can help you simplify the experiments with different architectures, losses, and optimizers. Let's define a
:class:`argus.model.Model` with two models via a dictionary. If you want to use PyTorch losses and optimizers, it's not
necessary to define them in the argus model.

.. code-block:: python

    from torchvision.models import resnet18

    class FlexModel(argus.Model):
        nn_module = {
            'net': Net,
            'resnet18': resnet18
        }


Create a model instance. Parameters for nn_module is a tuple where the first element is a name, second is init arguments.
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


Argus allows managing different parts combinations of a pipeline.

.. code-block:: python

    class MoreFlexModel(argus.Model):
        nn_module = {
            'net': Net,
            'resnet18': resnet18
        }
        optimizer = {
            'SGD': torch.optim.SGD,
            'adam_w': torch.optim.AdamW
        }
        loss = {
            'BCE': nn.BCEWithLogitsLoss,
            'cross_entropy': nn.CrossEntropyLoss,
            'nll': nn.NLLLoss
        }
        prediction_transform = {
            'sigmoid': nn.Sigmoid,
            'Softmax': nn.Softmax
        }


    params = {
        'nn_module': ('resnet18', {
            'pretrained': False,
            'num_classes': 1
        }),
        'optimizer': ('adam_w', {
            'lr': 0.01,
            'weight_decay': 0.042
        }),
        'loss': ('BCE', {'reduction': 'sum'}),
        'prediction_transform': 'sigmoid',
        'device': 'cuda'
    }

    model = MoreFlexModel(params)

.. seealso::
    If you need more flexibility you can:

    * Override methods of :class:`argus.model.Model`. For example, :meth:`argus.model.Model.train_step`
      and :meth:`argus.model.Model.val_step`. See :ref:`train_and_val_steps` guide for details.
    * Create a custom :class:`argus.callbacks.Callback`. See :ref:`custom_callbacks` guide.
    * Implement a custom :class:`argus.metrics.Metric`. See :ref:`custom_metrics` guide.
