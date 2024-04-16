Guides
======

The guides provide an in-depth overview of how the argus framework works and how one could customize it for specific needs.

.. toctree::
    :maxdepth: 2


.. _train_and_val_steps:

Train and val steps
-------------------

:meth:`argus.model.Model.train_step` and :meth:`argus.model.Model.val_step` are essential building blocks for training pipelines.
The methods are responsible for processing a single batch during the training or validation loop iterations.
This section describes the internals of these methods and provides some hints on
how to customize them if the defaults are not suitable for the desired application.

:meth:`argus.model.Model.train_step` performs the following steps on each batch:

1. Set the main model :class:`torch.nn.Module` into the training mode
   (see :meth:`torch.nn.Module.train`) if it is not already.
2. Move the batch data (inputs and targets) to the desired device, such as `cuda:0`.
3. Perform a forward pass, compute the loss function value and perform a backward pass.
4. Update the neural network weights.
5. Prepare the batch output, including the `prediction_transform` application, as described below.

:meth:`argus.model.Model.val_step` works quite in the same way, but without gradients
computation and weights update:

1. Set the main model :class:`torch.nn.Module` into the evaluation mode
   (see :meth:`torch.nn.Module.train`) if it is not already.
2. Move the batch data (inputs and targets) to the desired device, such as `cuda:0`.
3. Make a prediction on the provided input data, compute the loss function value.
4. Prepare the batch output, including the `prediction_transform` application, as described below.

The return value of `train_step`, as well as `val_step`, is a dictionary with the following structure:

* **"prediction"** - The model predictions on the batch samples. A `prediction_transform`
  function treats the predictions ahead of output if the function is presented
  in the :class:`argus.model.Model`. In the most basic scenario, the predictions
  are just a :class:`torch.Tensor` output of the model on the device used for processing.
  However, the `prediction_transform` could arbitrarily modify the data, including
  data type conversion.
* **"target"** - The target values for the batch samples. The data will be returned as a :class:`torch.Tensor` on the device used for the batch processing.
* **"loss"** - The loss function value, obtained as *loss.item()*.

The output structure above is good to know because it is used as a
:class:`argus.metrics.Metric` input and it needs to be parsed in the case of
a custom metric.

Customization
^^^^^^^^^^^^^

These step functions could be purposely customized.
For example, one may change the `train_step` to utilize `mixed precision <https://pytorch.org/docs/stable/notes/amp_examples.html>`_ training
or to apply a batch accumulation technique.
It is convenient to use the original implementation as a reference.

**Example**

A simple model example shows how to modify the `train_step` to employ automatic mixed precision training.

.. code-block:: python

    import torch
    import torchvision
    from argus import Model
    from argus.utils import deep_to, deep_detach


    class AMPModel(Model):
        nn_module = torchvision.models.resnet18
        loss = torch.nn.CrossEntropyLoss
        optimizer = torch.optim.SGD

        def __init__(self, params):
            super().__init__(params)
            self.scaler = torch.cuda.amp.GradScaler()

        def train_step(self, batch, state) -> dict:
            self.train()
            self.optimizer.zero_grad()
            input, target = deep_to(batch, device=self.device, non_blocking=True)
            # Custom part of a train step
            with torch.cuda.amp.autocast(enabled=True):
                prediction = self.nn_module(input)
                loss = self.loss(prediction, target)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # End of the custom code

            prediction = deep_detach(prediction)
            target = deep_detach(target)
            prediction = self.prediction_transform(prediction)
            return {
                'prediction': prediction,
                'target': target,
                'loss': loss.item()
            }


    params = {
        'nn_module': {'num_classes': 10},
        'optimizer': {'lr': 0.001},
        'device': 'cuda:0'
    }
    model = AMPModel(params)

The code creates a model, which allows training ResNet18 on a 10-class image
classification task with AMP.

For details on mixed precision training see PyTorch 
`tutorials <https://pytorch.org/docs/stable/notes/amp_examples.html>`_.
More Argus `train_step` and `val_step` customization cases could be found in :doc:`examples`.


.. note::
  :meth:`argus.model.Model.train_step` and :meth:`argus.model.Model.val_step` are
  independent of each other. Customization of either function does not lead to
  alternation of the second one.


.. _advanced_model_loading:

Advanced model loading
----------------------

An argus model could be saved with :meth:`argus.model.Model.save` or with help of an
:class:`argus.callbacks.Callback`, such as :class:`argus.callbacks.Checkpoint` or :class:`argus.callbacks.MonitorCheckpoint`.

:func:`argus.model.load_model` provides flexible interface to load a saved argus model.
The simplest user case is allows to load a model with saved parameters and components.

.. code:: python

    from argus import load_model

    # Argus model class should correspond to the model file to load.
    import ArgusModelClass


    model = load_model('/path/to/model/file')

However, the model loading process may require customizations; some cases are provided below.

1. Load the model to a specified device.
    Just provide the desired device name or a list of devices.

    .. code:: python

        # Load the model to cuda:0 device
        model = load_model('/path/to/model/file', device='cuda:0')

    The feature is helpful if one wants to load the model to a specific device for training or inference
    and also to load the model on a machine that does not have the device, which was specified before the
    model file was saved. For example, if the model was saved with ``device=='cuda:1'``,
    while the target machine is equipped with the only GPU, so, ``device=='cuda:0'`` is the only valid option.

    .. note::

        The feature allows to set the device for :class:`torch.nn.Module` model components only, i.e. ``nn_module`` and ``loss``.
        However, one should explicitly set the device for other device-dependent components, such as a
        ``prediction_transform`` requiring a device specification. See details in the cases below.


2. Load only some of the model components.
    It is possible to exclude ``loss``, ``optimizer`` or ``prediction_transform`` at
    the model load time if one or more components are not required. For example,
    it could be helpful for inference or if the component's code is not available.
    It is necessary to set the appropriate arguments to ``None`` to do this.

    .. code:: python

        # Load the model without optimizer and loss
        model = load_model('/path/to/model/file', loss=None, optimizer=None)

3. Alternate a model component parameters.
    ``nn_module``, ``loss``, ``optimizer`` or ``prediction_transform`` parameters could
    be customized during the model loading. Appropriate arguments should be set to parameters dicts to do this.

    .. code:: python

        # The prediction transform class of the model should accept `device` argument on creation

        # Load the model to 'cuda:1' device and also set the prediction_transform
        # to the correct device
        my_device = 'cuda:1'
        model = load_model('/path/to/model/file', prediction_transform={'device': my_device},
                           device=my_device)

.. seealso::
    * For more information see the :func:`argus.model.load_model` documentation.
    * More real-world examples of how to use `load_model` are available
      `here <https://github.com/lRomul/argus/blob/master/examples/load_model.py>`_.


.. _custom metrics:

Custom metrics
--------------


A custom metric can be implemented as a class, inheriting from 
:class:`argus.metrics.Metric` and redefining the following methods as required:


* :meth:`argus.metrics.Metric.reset`: Initialization or reset of the internal variables and accumulators.
  Normally, the method is called authomatically before each epoch start.

* :meth:`argus.metrics.Metric.update`: Update of the internal variables and accumulators based on the provided
  ``step_output`` results for a single step.
  The ``step_output`` is a dictionary containing the predictions, targets, and loss
  value or other values as the result of a single :meth:`argus.model.Model.train_step` or :meth:`argus.model.Model.val_step`.
  The method is called for each step in the loop. The metric will be evaluated
  with the results of validation steps if ``val_loader`` was provided to :meth:`argus.model.Model.fit`.
  The metric is evaluated with the results of training steps in case ``metrics_on_train=True`` in :meth:`argus.model.Model.fit`.
  In the case both are enabled, the metrics will be computed independently for train and validation steps.

* :meth:`argus.metrics.Metric.compute`: Computes and returns a metric value based on
  the accumulated values. The method is called at the end of an epoch to obtain the
  final metric value. Normally, the return of the method is a single float value. The value is reported
  in the logs with the metric name and prefixes to indicate the stage (train or val) and the metric name itself.
  For example, the results of a metric with name ``f1`` can be assesed as ``train_f1`` and ``val_f1``.
  The metric name can be used to assign callbacks actions, such as use as ``monitor`` for
  :class:`argus.callbacks.MonitorCheckpoint`.

The :class:`argus.metrics.Metric` base class initialization requires two attributes: ``name`` and ``better``.
The first attribute specifies the name of the evaluation metric, while the second indicates whether a higher value
(`max`) or a lower value (`min`) means improvement for this metric.

The code below demonstrates a top-K accuracy metric, which implements the required methods.

.. code-block:: python

    from argus.metrics import Metric


    class TopKAccuracy(Metric):
        """Calculate the top-K accuracy for multiclass classification.

        Args:
            k (int): Number of top predictions to consider.
        """

        name = 'top_k_accuracy'
        better = 'max'

        def __init__(self, k: int = 5):
            self.k = k
            self.correct = 0
            self.count = 0
            # Parametrized name allows having several instances of the metric with different k values
            self.name = f'top_{self.k}_accuracy'

        def reset(self):
            self.correct = 0
            self.count = 0

        def update(self, step_output: dict):
            indices = torch.topk(step_output['prediction'], k=self.k, dim=1)[1]
            target = step_output['target'].unsqueeze(1)
            correct = torch.any(indices == target, dim=1)
            self.correct += torch.sum(correct).item()
            self.count += correct.shape[0]

        def compute(self) -> float:
            if self.count == 0:
                raise RuntimeError('Must be at least one example for computation')
            return self.correct / self.count


In some more advanced use cases, it may be required to create a custom metric to
report not only a single value but several additional values. That can be the case
when the intermediate computed results are of interest for monitoring. This can
be useful to optimise the metric compute costs and memory usage. In order to do this, one should redefine 
:meth:`argus.metrics.Metric.epoch_complete` method. This method is called at the end of each epoch to update the model state
with all the metrics values. All the values to report should be added to ``state.metrics`` dictionary, using distinctive value
names as keys. The names prefix of the train stage is avalable in ``state.phase``

The example below shows a modified top-K accuracy metric, which reports not only a top-K
accuracy but also the average rank of the correct prediction for cases where the
correct answer was present among the top-K predictions.

.. code-block:: python

    from argus.engine import State
    from argus.metrics import Metric


    class TopKAccuracy(Metric):
        """Calculate the top-K accuracy for multiclass classification.

        It also reports the average rank of the correct top-K predictions.

        Args:
            k (int): Number of top predictions to consider.
        """
        name = 'top_k_accuracy'
        better = 'max'

        def __init__(self, k: int = 5):
            self.k = k
            self.correct = 0
            self.rank = 0
            self.count = 0
            self.name = f'top_{self.k}_accuracy'

        def reset(self):
            self.correct = 0
            self.rank = 0
            self.count = 0

        def update(self, step_output: dict):
            indices = torch.topk(step_output['prediction'], k=self.k, dim=1)[1]
            target = step_output['target'].unsqueeze(1)
            correct = torch.any(indices == target, dim=1)
            rank = torch.nonzero(indices == target)[:, 1]
            self.correct += torch.sum(correct).item()
            self.rank += torch.sum(rank).item()
            self.count += correct.shape[0]

        def compute(self) -> float:
            if self.count == 0:
                raise RuntimeError('Must be at least one example for computation')
            return self.correct / self.count

        def epoch_complete(self, state: State):
            with torch.no_grad():
                accuracy = self.compute()
            rank = self.rank / self.count + 1.0  # +1.0 because ranks are 1-indexed
            name_prefix = f"{state.phase}_" if state.phase else ''
            state.metrics[f'{name_prefix}{self.name}'] = accuracy
            state.metrics[f'{name_prefix}rank_{self.k}'] = rank