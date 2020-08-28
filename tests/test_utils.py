import torch
import pytest

from argus.utils import (
    Default,
    Identity,
    deep_to,
    deep_detach,
    deep_chunk,
    device_to_str,
    check_pickleble,
    AverageMeter
)


@pytest.fixture(params=[(16, torch.float16),
                        (37, torch.float32)])
def list_of_tensors(request):
    first_dim, dtype = request.param
    return [
        torch.zeros(first_dim, dtype=dtype, requires_grad=True),
        torch.ones(first_dim, 4, dtype=dtype, requires_grad=True),
        torch.randint(42, size=(first_dim, 4, 2),
                      dtype=dtype, requires_grad=True)
    ]


@pytest.fixture
def dict_of_tensors(list_of_tensors):
    return {str(i): tensor for i, tensor in enumerate(list_of_tensors)}


def test_default():
    default = Default()
    assert "default" == str(default)


@pytest.mark.parametrize("x", [42, 'test', True])
def test_identity(x):
    identity = Identity()
    assert x == identity(x)
    assert "Identity()" == str(identity)


@pytest.mark.parametrize("destination_dtype", [torch.float16, torch.float32])
def test_deep_to(list_of_tensors, dict_of_tensors, destination_dtype):

    output_list = deep_to(list_of_tensors, dtype=destination_dtype)
    assert all([tensor.dtype == destination_dtype for tensor in output_list])

    output_dict = deep_to(dict_of_tensors, dtype=destination_dtype)
    assert all([isinstance(key, str) for key in output_dict.keys()])
    assert all([tensor.dtype == destination_dtype for tensor in output_dict.values()])

    nn_module = torch.nn.Linear(128, 8)
    output_nn_module = deep_to(nn_module, dtype=destination_dtype)
    assert output_nn_module.weight.dtype == destination_dtype

    assert 'qwerty' == deep_to('qwerty', dtype=destination_dtype)
    assert None is deep_to(None, dtype=destination_dtype)
    assert deep_to(True, dtype=destination_dtype)


def test_deep_detach(list_of_tensors, dict_of_tensors):
    def all_grad_is_none(sequence):
        return all([tensor.grad is None for tensor in sequence])

    assert all_grad_is_none(list_of_tensors)
    assert all_grad_is_none(dict_of_tensors.values())

    list_of_grad_tensors = [tensor * 2 for tensor in list_of_tensors]
    dict_of_grad_tensors = {key: tensor * 2 for key, tensor in dict_of_tensors.items()}
    loss = torch.tensor(0.)
    for tensor in [*list_of_grad_tensors, *dict_of_grad_tensors.values()]:
        loss += tensor.sum()
    loss.backward()

    assert all_grad_is_none(deep_detach(list_of_tensors))
    assert all_grad_is_none(deep_detach(dict_of_tensors).values())

    assert 'qwerty' == deep_detach('qwerty')
    assert None is deep_detach(None)
    assert deep_detach(True)


def test_deep_chunk(list_of_tensors, dict_of_tensors):
    list_of_chunks = deep_chunk(list_of_tensors, 4)
    assert len(list_of_chunks) == 4
    for i, tensor in enumerate(list_of_tensors):
        sum_among_chunks = sum(c[i].shape[0] for c in list_of_chunks)
        assert sum_among_chunks == tensor.shape[0]

    list_of_dict_chunks = deep_chunk(dict_of_tensors, 4)
    assert len(list_of_dict_chunks) == 4
    for key, tensor in dict_of_tensors.items():
        sum_among_chunks = sum(c[key].shape[0] for c in list_of_dict_chunks)
        assert sum_among_chunks == tensor.shape[0]

    assert ['qwerty', 'qwerty'] == deep_chunk('qwerty', 2)
    assert [True, True] == deep_chunk(True, 2)


def test_device_to_str():
    assert 'cpu' == device_to_str(torch.device('cpu'))
    devices = torch.device('cuda:0'), torch.device('cuda:1')
    assert ['cuda:0', 'cuda:1'] == device_to_str(devices)


def test_check_pickleble(dict_of_tensors):
    check_pickleble(dict_of_tensors)
    with pytest.raises(TypeError):
        check_pickleble(pytest)


@pytest.mark.parametrize("values",
                         [list(range(42)),
                          torch.randint(1000, size=(42,)).tolist(),
                          (1e6 * torch.rand(dtype=torch.float32,
                                            size=(42,))).tolist()])
def test_average_meter(values):
    average_meter = AverageMeter()
    for value in values:
        average_meter.update(value)

    average = sum(values) / len(values)
    assert pytest.approx(average_meter.average) == average
