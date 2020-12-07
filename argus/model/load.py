import os
from typing import Union, Optional, Callable

import torch

from argus import types
from argus.model.build import MODEL_REGISTRY, cast_device
from argus.utils import deep_to, device_to_str, Default, default, identity


def default_change_state_dict_func(nn_state_dict: dict,
                                   optimizer_state_dict: Optional[dict] = None):
    return nn_state_dict, optimizer_state_dict


def load_model(file_path: types.Path,
               nn_module: Union[Default, types.Param] = default,
               optimizer: Union[Default, None, types.Param] = default,
               loss: Union[Default, None, types.Param] = default,
               prediction_transform: Union[Default, None, types.Param] = default,
               device: Union[Default, types.InputDevices] = default,
               change_params_func: Callable = identity,
               change_state_dict_func: Callable = default_change_state_dict_func,
               model_name: Union[Default, str] = default,
               **kwargs):
    """Load an argus model from a file.

    The function allows loading an argus model, saved with
    :meth:`argus.model.Model.save`. The model is always loaded in *eval* mode.

    Args:
        file_path (str or :class:`pathlib.Path`): Path to the file to load.
        device (str, torch.device or list of devices, optional): Device for the model.
        nn_module (dict, tuple or str, optional): Params of the nn_module to
            replace params in the state.
        optimizer (None, dict, tuple or str, optional): Params of the optimizer to
            replace params in the state. Optimizer is not created in the loaded
            model if it is set to `None`.
        loss (None, dict, tuple or str, optional): Params of the loss to replace params
            in the state. Loss is not created in the loaded model if it is set
            to `None`.
        prediction_transform (None, dict, tuple or str, optional): Params of the
            prediction_transform to replace params in the state.
            prediction_transform is not created in the loaded model if it is
            set to `None`.
        change_params_func (function, optional): Function for modification of
            the loaded params. It takes params from the loaded state as an
            input and outputs params to use during the model creation.
        change_state_dict_func (function, optional): Function for modification of
            nn_module and optimizer state dict. Takes `nn_state_dict` and
            `optimizer_state_dict` as inputs and outputs state dicts for the
            model creation.
        model_name (str, optional): Class name of :class:`argus.model.Model`.
            By default uses the name from the loaded state.

    Returns:
        :class:`argus.model.Model`: Loaded argus model.

    Example:

        .. code-block:: python

            model = ArgusModel(params)
            model.save(model_path, optimizer_state=True)

            # restarting python...

            # ArgusModel class must be in scope at this moment
            model = argus.load_model(model_path, device="cuda:0")

        More options how to use load_model you can find
        `here <https://github.com/lRomul/argus/blob/master/examples/load_model.py>`_.

    Raises:
        ImportError: If the model is not available in the scope. Often it means
            that it is not imported or defined.
        FileNotFoundError: If the file is not found by the *file_path*.

    """

    if os.path.isfile(file_path):
        state = torch.load(file_path)

        if isinstance(model_name, Default):
            str_model_name = state['model_name']
        else:
            str_model_name = model_name

        if str_model_name in MODEL_REGISTRY:
            params = state['params']
            if not isinstance(device, Default):
                params['device'] = device_to_str(cast_device(device))

            if nn_module is not default:
                if nn_module is None:
                    raise ValueError("nn_module is required attribute for argus.Model")
                params['nn_module'] = nn_module
            if optimizer is not default:
                params['optimizer'] = optimizer
            if loss is not default:
                params['loss'] = loss
            if prediction_transform is not default:
                params['prediction_transform'] = prediction_transform

            for attribute, attribute_params in kwargs.items():
                params[attribute] = attribute_params

            model_class = MODEL_REGISTRY[str_model_name]
            params = change_params_func(params)
            model = model_class(params)
            nn_state_dict = deep_to(state['nn_state_dict'], model.device)
            optimizer_state_dict = None
            if 'optimizer_state_dict' in state:
                optimizer_state_dict = deep_to(state['optimizer_state_dict'], model.device)
            nn_state_dict, optimizer_state_dict = change_state_dict_func(nn_state_dict,
                                                                         optimizer_state_dict)

            model.get_nn_module().load_state_dict(nn_state_dict)
            if model.optimizer is not None and optimizer_state_dict is not None:
                model.optimizer.load_state_dict(optimizer_state_dict)
            model.eval()
            return model
        else:
            raise ImportError(f"Model '{model_name}' not found in scope")
    else:
        raise FileNotFoundError(f"No state found at {file_path}")
