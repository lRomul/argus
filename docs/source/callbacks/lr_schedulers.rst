Learning rate schedulers
========================

Callbacks for auto adjust the learning rate based on the number of epochs or other metrics measurements.

The learning rates schedulers allow implementing dynamic learning rate changing policy.
These callbacks are wrappers of native PyTorch `torch.optim.lr_scheduler`.

Currently, the following schedulers are available (see PyTorch documentation by the links provided for details on the schedulers algorithms themself):

.. currentmodule:: argus.callbacks

LambdaLR
--------
.. autoclass:: LambdaLR
   :members:

`PyTorch docs on LambdaLR <https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.LambdaLR>`_

StepLR
------
.. autoclass:: StepLR
   :members:

`PyTorch docs on StepLR <https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.StepLR>`_


MultiStepLR
-----------
.. autoclass:: MultiStepLR
   :members:

* `PyTorch docs on MultiStepLR <https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.MultiStepLR>`_

ExponentialLR
-------------
.. autoclass:: ExponentialLR
   :members:

* `PyTorch docs on ExponentialLR <https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ExponentialLR>`_

CosineAnnealingLR
-----------------
.. autoclass:: CosineAnnealingLR
   :members:

* `PyTorch docs on CosineAnnealingLR <https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.CosineAnnealingLR>`_

ReduceLROnPlateau
-----------------
.. autoclass:: ReduceLROnPlateau
   :members:

* `PyTorch docs on ReduceLROnPlateau <https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau>`_

CyclicLR
--------
.. autoclass:: CyclicLR
   :members:

* `PyTorch docs on CyclicLR <https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.CyclicLR>`_

CosineAnnealingWarmRestarts
---------------------------
.. autoclass:: CosineAnnealingWarmRestarts
   :members:

* `PyTorch docs on CosineAnnealingWarmRestarts <https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts>`_

MultiplicativeLR
----------------
.. autoclass:: MultiplicativeLR
   :members:

* `PyTorch docs on MultiplicativeLR <https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.MultiplicativeLR>`_

OneCycleLR
----------
.. autoclass:: OneCycleLR
   :members:

* `PyTorch docs on OneCycleLR <https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.OneCycleLR>`_
