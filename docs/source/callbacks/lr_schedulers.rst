Learning rate schedulers
========================

Callbacks for auto adjust the learning rate based on the number of epochs or other metrics measurements.

The learning rates schedulers allow implementing dynamic learning rate changing policy.
These callbacks are wrappers of native PyTorch :mod:`torch.optim.lr_scheduler`.

Currently, the following schedulers are available (see PyTorch documentation by the links provided for details on the
schedulers algorithms themself):

.. currentmodule:: argus.callbacks

LambdaLR
--------
.. autoclass:: LambdaLR
   :members:

PyTorch docs on :class:`torch.optim.lr_scheduler.LambdaLR`.

StepLR
------
.. autoclass:: StepLR
   :members:

PyTorch docs on :class:`torch.optim.lr_scheduler.StepLR`.


MultiStepLR
-----------
.. autoclass:: MultiStepLR
   :members:

PyTorch docs on :class:`torch.optim.lr_scheduler.MultiStepLR`.

ExponentialLR
-------------
.. autoclass:: ExponentialLR
   :members:

PyTorch docs on :class:`torch.optim.lr_scheduler.ExponentialLR`.

CosineAnnealingLR
-----------------
.. autoclass:: CosineAnnealingLR
   :members:

PyTorch docs on :class:`torch.optim.lr_scheduler.CosineAnnealingLR`.

ReduceLROnPlateau
-----------------
.. autoclass:: ReduceLROnPlateau
   :members:

PyTorch docs on :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`.

CyclicLR
--------
.. autoclass:: CyclicLR
   :members:

PyTorch docs on :class:`torch.optim.lr_scheduler.CyclicLR`.

CosineAnnealingWarmRestarts
---------------------------
.. autoclass:: CosineAnnealingWarmRestarts
   :members:

PyTorch docs on :class:`torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`.

MultiplicativeLR
----------------
.. autoclass:: MultiplicativeLR
   :members:

PyTorch docs on :class:`torch.optim.lr_scheduler.MultiplicativeLR`.

OneCycleLR
----------
.. autoclass:: OneCycleLR
   :members:

PyTorch docs on :class:`torch.optim.lr_scheduler.OneCycleLR`.
