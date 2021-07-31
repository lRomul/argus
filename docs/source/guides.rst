Guides
======

The guides provide an in-depth overview of how the argus framework works and how one could customize it for specific needs.

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
:class:`argus.callbacks.Callback` input and it needs to be parsed in the case of
a custom Callback.

Customization
^^^^^^^^^^^^^

These step functions could be purposely customized.
For example, one may change the `train_step` to utilize `mixed precision <https://pytorch.org/docs/stable/notes/amp_examples.html>`_ training
or to apply a batch accumulation technique.
It is convenient to use the original implementation as a reference.

*Note:* :meth:`argus.model.Model.train_step` and :meth:`argus.model.Model.val_step` are independent of each other.
Customization of either function does not lead to alternation of the second one.
