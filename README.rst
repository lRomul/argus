Argus
=====

Argus is easy-to-use flexible library for training neural networks in PyTorch.


Warning
=======
The project is in development, so it is not yet suitable for use.


Roadmap
=======
* Save and load models :heavy_check_mark:
* Improve event handlers (attach callbacks) :heavy_check_mark:
* ModelCheckpoint, EarlyStopping :heavy_check_mark:
* LR schedulers :heavy_check_mark:
* DataParallel for multi-gpu training :heavy_check_mark:
* More informative README
* More examples (imagenet, pytorch-cnn-finetune)
* More metrics
* Improve error handling
* Code refactoring
* Docs
* Tests
* Co-training multiple models (?)


Installation
============

From pip:

.. code:: bash

    pip install pytorch-argus

From source:

.. code:: bash

    python setup.py install


Examples
========

Full MNIST example you can see `here <https://github.com/lRomul/argus/blob/master/examples/mnist.py>`_.
MNIST VAE `example <https://github.com/lRomul/argus/blob/master/examples/mnist_vae.py>`_.

.. code-block:: python

    import torch
    from torch import nn
    import torch.nn.functional as F
    from mnist_utils import get_data_loaders

    from argus import Model, load_model
    from argus.callbacks import MonitorCheckpoint, EarlyStopping, ReduceLROnPlateau


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

        callbacks = [
            MonitorCheckpoint(dir_path='mnist', monitor='val_accuracy', max_saves=3),
            EarlyStopping(monitor='val_accuracy', patience=9),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3)
        ]

        model.fit(train_loader,
                  val_loader=val_loader,
                  max_epochs=50,
                  metrics=['accuracy'],
                  callbacks=callbacks,
                  metrics_on_train=True)

        del model
        model = load_model('mnist/model-last.pth')


You can use Argus with ``make_model`` from `pytorch-cnn-finetune <https://github.com/creafz/pytorch-cnn-finetune>`_.

.. code-block:: python

    from cnn_finetune import make_model
    from argus import Model

    class CnnFinetune(Model):
        nn_module = make_model


    params = {
        'nn_module': {
            'model_name': 'resnet18',
            'num_classes': 10,
            'pretrained': False,
            'input_size': (256, 256)
        },
        'optimizer': ('Adam', {'lr': 0.01}),
        'loss': 'CrossEntropyLoss',
        'device': 'cuda'
    }

    model = CnnFinetune(params)
    
Full Argus pipeline of 14th place solution for Kaggle TGS Salt Identification Challenge. `link <https://github.com/lRomul/argus-tgs-salt>`_
