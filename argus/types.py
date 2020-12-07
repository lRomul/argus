import torch
import pathlib
from typing import Any, Union, Tuple, List, Dict

from argus.callbacks import Callback
from argus.metrics import Metric

Path = Union[pathlib.Path, str]

InputDevices = Union[str, torch.device, List[Union[str, torch.device]]]
Devices = Union[torch.device, List[torch.device]]
AttrMeta = Union[Dict[str, Any], Any]
Param = Union[dict, Tuple[str, dict]]

Callbacks = List[Callback]
Metrics = List[Union[Metric, str]]
