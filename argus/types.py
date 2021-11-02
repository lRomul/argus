import torch
import pathlib
from typing import Any, Union, Tuple, List, Dict, TypeVar

__all__ = ["Path", "InputDevices", "Devices", "AttrMeta", "Param"]


TVar = TypeVar('TVar')

Path = Union[pathlib.Path, str]

InputDevices = Union[str, torch.device, List[Union[str, torch.device]]]
Devices = Union[torch.device, List[torch.device]]
AttrMeta = Union[Dict[str, Any], Any]
Param = Union[dict, Tuple[str, dict]]
