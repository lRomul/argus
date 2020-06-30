import os
import numbers

import torch

from argus.model.build import BuildModel, MODEL_REGISTRY, cast_device
from argus.engine import Engine, Events
from argus.utils import deep_to, deep_detach, device_to_str
from argus.callbacks import Callback, on_epoch_complete
from argus.callbacks.logging import metrics_logging
from argus.metrics.metric import Metric, METRIC_REGISTRY
from argus.metrics.loss import Loss


def _attach_callbacks(engine, callbacks):
    if callbacks is not None:
        for callback in callbacks:
            if isinstance(callback, Callback):
                callback.attach(engine)
            else:
                raise TypeError(
                    f"Expected callback type {Callback}, got {type(callback)}"
                )


def _attach_metrics(engine, metrics, name_prefix=''):
    for metric in metrics:
        if isinstance(metric, str):
            if metric in METRIC_REGISTRY:
                metric = METRIC_REGISTRY[metric]()
            else:
                raise ValueError(f"Metric '{metric}' not found in scope")
        if isinstance(metric, Metric):
            metric.attach(engine, {
                Events.EPOCH_COMPLETE: {'name_prefix': name_prefix}
            })
        else:
            raise TypeError(
                f"Expected metric type {Metric} or str, got {type(metric)}"
            )


class Model(BuildModel):
    """Model
    """

    def __init__(self, params: dict):
        super().__init__(params)

    def train_step(self, batch, state) -> dict:
        """Perform a single train step.

        The method is used by :class:`argus.engine.Engine`.
        The train step includes input and target tensor transition to the model
        device, forward pass, loss evaluation, backward pass, and the train
        batch prediction preparation with a *prediction_transform*.

        Args:
            batch (tuple of 2 torch.Tensors: (input, target)): The input data
                and target tensors to process.
            state (:class:`argus.engine.State`): The argus model state.

        Returns:
            dict: The train step results::

                {
                    'prediction': The train batch predictions,
                    'target': The train batch target data on the model device,
                    'loss': Loss function value
                }

        """
        self.train()
        self.optimizer.zero_grad()
        input, target = deep_to(batch, device=self.device, non_blocking=True)
        prediction = self.nn_module(input)
        loss = self.loss(prediction, target)
        loss.backward()
        self.optimizer.step()

        prediction = deep_detach(prediction)
        target = deep_detach(target)
        prediction = self.prediction_transform(prediction)
        return {
            'prediction': prediction,
            'target': target,
            'loss': loss.item()
        }

    def val_step(self, batch, state) -> dict:
        """Perform a single validation step.

        The method is used by :class:`argus.engine.Engine`.
        The validation step includes input and target tensor transition to the
        model device, forward pass, loss evaluation, and the train batch
        prediction preparation with a *prediction_transform*.

        Gradient calculations and the model weights update are omitted, which
        is the main difference with the :meth:`train_step`
        method.

        Args:
            batch (tuple of 2 torch.Tensors: (input, target)): The input data
                and target tensors to process.
            state (:class:`argus.engine.State`): The argus model state.

        Returns:
            dict: The train step results::

                {
                    'prediction': The train batch predictions,
                    'target': The train batch target data on the model device,
                    'loss': Loss function value
                }

        """
        self.eval()
        with torch.no_grad():
            input, target = deep_to(batch, device=self.device, non_blocking=True)
            prediction = self.nn_module(input)
            loss = self.loss(prediction, target)
            prediction = self.prediction_transform(prediction)
            return {
                'prediction': prediction,
                'target': target,
                'loss': loss.item()
            }

    def fit(self,
            train_loader,
            val_loader=None,
            num_epochs=1,
            metrics=None,
            metrics_on_train=False,
            callbacks=None,
            val_callbacks=None):
        """Train the argus model.

        The method attaches metrics and callbacks to the train and validation,
        and runs the training process.

        Args:
            train_loader (torch.utils.data.DataLoader): The train dataloader.
            val_loader (torch.utils.data.DataLoader or `None`, optional):
                The validation dataloader. Defaults to `None`.
            num_epochs (int, optional): Number of training epochs to
                run. Defaults to 1.
            metrics (list of :class:`argus.metrics.Metric` or `None`, optional):
                List of metrics to evaluate. By default, the metrics are
                evaluated on the validation data (if any) only.
                Defaults to `None`.
            metrics_on_train (bool, optional): Evaluate the metrics on train
                data as well. Defaults to False.
            callbacks (list of :class:`argus.callbacks.Callback` or `None`, optional):
                List of callbacks to be attached to the training process.
                Defaults to `None`.
            val_callbacks (list of :class:`argus.callbacks.Callback` or `None`, optional):
                List of callbacks to be attached to the validation process.
                Defaults to `None`.

        """
        assert self.train_ready()
        metrics = [] if metrics is None else metrics

        train_engine = Engine(self.train_step, model=self, logger=self.logger)
        train_metrics = [Loss()] + metrics if metrics_on_train else [Loss()]
        _attach_metrics(train_engine, train_metrics, name_prefix='train_')
        metrics_logging.attach(train_engine, train=True)

        if val_loader is not None:
            self.validate(val_loader, metrics, val_callbacks)
            val_engine = Engine(self.val_step, model=self, logger=self.logger)
            _attach_metrics(val_engine, [Loss()] + metrics, name_prefix='val_')
            _attach_callbacks(val_engine, val_callbacks)

            @on_epoch_complete
            def validation_epoch(train_state, val_engine, val_loader):
                epoch = train_state.epoch
                val_state = val_engine.run(val_loader, epoch, epoch + 1)
                train_state.metrics.update(val_state.metrics)

            validation_epoch.attach(train_engine, val_engine, val_loader)
            metrics_logging.attach(train_engine, train=False)

        _attach_callbacks(train_engine, callbacks)
        train_engine.run(train_loader, 0, num_epochs)

    def validate(self, val_loader, metrics=None, callbacks=None):
        """Perform a validation.

        Args:
            val_loader (torch.utils.data.DataLoader): The validation
                dataloader.
            metrics (list of :class:`argus.metrics.Metric` or `None`, optional):
                List of metrics to evaluate with the data. Defaults to `None`.
            callbacks (list of :class:`argus.callbacks.Callback` or `None`, optional):
                List of callbacks to be attached to the validation process.
                Defaults to `None`.

        Returns:
            dict: The metrics dictionary.

        """
        assert self.train_ready()
        metrics = [] if metrics is None else metrics
        val_engine = Engine(self.val_step, model=self, logger=self.logger)
        _attach_metrics(val_engine, [Loss()] + metrics, name_prefix='val_')
        _attach_callbacks(val_engine, callbacks)
        metrics_logging.attach(val_engine, train=False, print_epoch=False)
        return val_engine.run(val_loader).metrics

    def set_lr(self, lr):
        """Set the learning rate for the optimizer.

        The method allows setting individual learning rates for the optimizer
        parameter groups as well as setting even learning rate for all
        parameters.

        Args:
            lr (number or list/tuple of numbers): The learning rate to set. If
                a single number is provided, all parameter groups learning
                rates are set to the same value. In order to set individual
                learning rates for each parameter group, a list or tuple of
                values with the corresponding length should be provided.

        Raises:
            ValueError: If *lr* is a list or tuple and its length is not equal
                to the number of parameter groups.
            ValueError: If *lr* type is not list, tuple, or number.
            AttributeError: If the model is not *train_ready* (i.e. not all
                attributes are set).

        """
        assert self.train_ready()
        param_groups = self.optimizer.param_groups
        if isinstance(lr, (list, tuple)):
            lrs = list(lr)
            if len(lrs) != len(param_groups):
                raise ValueError(f"Expected lrs length {len(param_groups)}, "
                                 f"got {len(lrs)}")
        elif isinstance(lr, numbers.Number):
            lrs = [lr] * len(param_groups)
        else:
            raise ValueError(f"Expected lr type list, tuple or number, "
                             f"got {type(lr)}")
        for group_lr, param_group in zip(lrs, param_groups):
            param_group['lr'] = group_lr

    def get_lr(self):
        """Get the learning rate from the optimizer.

        It could be a single value or a list of values in the case of multiple
        parameter groups.

        Returns:
            (float or a list of floats): The learning rate value or a list of
            individual parameter groups learning rate values.

        """
        assert self.train_ready()
        lrs = []
        for param_group in self.optimizer.param_groups:
            lrs.append(param_group['lr'])
        if len(lrs) == 1:
            return lrs[0]
        return lrs

    def save(self, file_path):
        """Save the argus model into a file.

        The argus model is saved as a dict::

            {
                'model_name': Name of the argus model,
                'params': Argus model parameters dict,
                'nn_state_dict': torch nn_module.state_dict()
            }

        The *state_dict* is always transferred to cpu prior to saving.

        Args:
            file_path (str): Path to the argus model file.

        """
        nn_module = self.get_nn_module()
        state = {
            'model_name': self.__class__.__name__,
            'params': self.params,
            'nn_state_dict': deep_to(nn_module.state_dict(), 'cpu')
        }
        torch.save(state, file_path)
        self.logger.info(f"Model saved to '{file_path}'")

    def predict(self, input):
        """Make a prediction with the given input.

        The prediction process consists of the input tensor transferring to the
        model device, forward pass of the nn_module in *eval* mode and
        application of the prediction_transform to the raw prediction output.

        Args:
            input (torch.Tensor): The input tensor to predict with. It will be
                transferred to the model device. The user is responsible for
                ensuring that the input tensor shape and type match the model.

        Returns:
            torch.Tensor or other type: Predictions as the result of the
                prediction_transform application.

        """
        assert self.predict_ready()
        with torch.no_grad():
            self.eval()
            input = deep_to(input, self.device)
            prediction = self.nn_module(input)
            prediction = self.prediction_transform(prediction)
            return prediction

    def train(self):
        """Set the nn_module into train mode."""
        if not self.nn_module.training:
            self.nn_module.train()

    def eval(self):
        """Set the nn_module into eval mode."""
        if self.nn_module.training:
            self.nn_module.eval()


def load_model(file_path, device=None):
    """Load an argus model from a file.

    The function allows loading an argus model, saved with
    :meth:`argus.model.Model.save`. The model is always loaded in *eval* mode.

    Args:
        file_path (str): Path to the file to load.
        device (str, optional): Device for the model. Defaults to None.

    Raises:
        ImportError: If the model is not available in the scope. Often it means
            that it is not imported or defined.
        FileNotFoundError: If the file is not found by the *file_path*.

    Returns:
        :class:`argus.model.Model`: Loaded argus model.

    """
    if os.path.isfile(file_path):
        state = torch.load(file_path)

        if state['model_name'] in MODEL_REGISTRY:
            params = state['params']
            if device is not None:
                device = cast_device(device)
                device = device_to_str(device)
                params['device'] = device

            model_class = MODEL_REGISTRY[state['model_name']]
            model = model_class(params)
            nn_state_dict = deep_to(state['nn_state_dict'], model.device)

            model.get_nn_module().load_state_dict(nn_state_dict)
            model.eval()
            return model
        else:
            raise ImportError(f"Model '{state['model_name']}' not found in scope")
    else:
        raise FileNotFoundError(f"No state found at {file_path}")
