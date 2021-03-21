import pytest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from argus import load_model
from argus.model import Model
from argus.utils import Identity


@pytest.fixture(scope='function', params=[False, True])
def saved_argus_model(request, tmpdir, linear_argus_model_instance, get_batch_function):
    optimizer_state = request.param
    model = linear_argus_model_instance
    train_dataset = TensorDataset(*get_batch_function(batch_size=1024))
    train_loader = DataLoader(train_dataset, shuffle=True,
                              drop_last=True, batch_size=32)
    model.fit(train_loader, num_epochs=3)
    path = str(tmpdir.mkdir("experiment").join("model.pth"))
    model.save(path, optimizer_state=optimizer_state)
    return optimizer_state, path, model


def check_weights(model, loaded_model):
    nn_state_dict = model.nn_module.state_dict()
    for layer_name, weight in loaded_model.nn_module.state_dict().items():
        assert layer_name in nn_state_dict
        assert torch.all(nn_state_dict[layer_name] == weight)
    return True


class TestLoadModel:
    def test_load_model(self, saved_argus_model):
        optimizer_state, path, model = saved_argus_model
        loaded_model = load_model(path, device='cpu')

        assert loaded_model.params == model.params
        assert check_weights(model, loaded_model)
        assert isinstance(loaded_model.loss, model.loss.__class__)
        assert isinstance(loaded_model.optimizer, model.optimizer.__class__)
        assert isinstance(loaded_model.prediction_transform,
                          model.prediction_transform.__class__)

        nn.init.xavier_uniform_(loaded_model.nn_module.fc.weight)
        with pytest.raises(AssertionError):
            assert check_weights(model, loaded_model)

        if optimizer_state:
            assert torch.all(
                loaded_model.optimizer.state_dict()['state'][0]['momentum_buffer']
                == model.optimizer.state_dict()['state'][0]['momentum_buffer']
            )
        else:
            assert loaded_model.optimizer.state_dict()['state'] == {}

    def test_none_attributes(self, saved_argus_model):
        optimizer_state, path, model = saved_argus_model
        assert load_model(path, optimizer=None).optimizer is None
        assert load_model(path, loss=None).loss is None
        assert load_model(path, prediction_transform=None).prediction_transform is None
        with pytest.raises(ValueError):
            assert load_model(path, nn_module=None)

    def test_replace_nn_module_params(self, saved_argus_model):
        optimizer_state, path, model = saved_argus_model
        nn_module_params = ('LinearNet', {
            'in_features': 4,
            'out_features': 1,
            'sigmoid': True
        })
        loaded_model = load_model(path, nn_module=nn_module_params)
        assert loaded_model.nn_module.sigmoid
        assert loaded_model.params['nn_module'] == nn_module_params
        assert not model.nn_module.sigmoid

    def test_replace_optimizer_params(self, saved_argus_model):
        optimizer_state, path, model = saved_argus_model
        optimizer_params = ('Adam', {'lr': 0.42})
        loaded_model = load_model(path, optimizer=optimizer_params,
                                  change_state_dict_func=lambda nn, optim: (nn, None))
        assert isinstance(loaded_model.optimizer, torch.optim.Adam)
        assert loaded_model.get_lr() == 0.42
        assert loaded_model.params['optimizer'] == optimizer_params
        assert not isinstance(model.optimizer, torch.optim.Adam)

    def test_replace_loss_params(self, saved_argus_model):
        optimizer_state, path, model = saved_argus_model
        loaded_model = load_model(path, loss='BCEWithLogitsLoss')
        assert isinstance(loaded_model.loss, nn.BCEWithLogitsLoss)
        assert loaded_model.params['loss'] == 'BCEWithLogitsLoss'
        assert not isinstance(model.loss, nn.BCEWithLogitsLoss)

    def test_replace_prediction_transform_params(self, saved_argus_model):
        optimizer_state, path, model = saved_argus_model
        loaded_model = load_model(path, prediction_transform='Sigmoid')
        assert isinstance(loaded_model.prediction_transform, nn.Sigmoid)
        assert loaded_model.params['prediction_transform'] == 'Sigmoid'
        assert not isinstance(model.prediction_transform, nn.Sigmoid)

    def test_replace_kwargs_params(self, saved_argus_model):
        optimizer_state, path, model = saved_argus_model
        loaded_model = load_model(path, new_param={"qwerty": 42})
        assert loaded_model.params['new_param'] == {"qwerty": 42}

    def test_file_not_found_error(self):
        with pytest.raises(FileNotFoundError):
            load_model('/fake/path/to/nothing.pth')

    def test_replace_model_name(self,
                                saved_argus_model,
                                linear_net_class,
                                vision_net_class):
        optimizer_state, path, model = saved_argus_model

        if optimizer_state:
            return

        class ArgusReplaceModel(Model):
            nn_module = {
                'LinearNet': linear_net_class,
                'VisionNet': vision_net_class
            }
            prediction_transform = {
                'Sigmoid': nn.Sigmoid,
                'Identity': Identity
            }

        loaded_model = load_model(path, model_name='ArgusReplaceModel')
        assert isinstance(loaded_model, ArgusReplaceModel)
        assert not isinstance(loaded_model, model.__class__)

        with pytest.raises(ImportError):
            load_model(path, model_name='Qwerty')

    def test_state_load_func(self, saved_argus_model):
        def custom_state_load_func(file_path):
            file_path = Path(file_path) / 'model.pth'
            return torch.load(file_path)

        optimizer_state, path, model = saved_argus_model
        loaded_model = load_model(Path(path).parent,
                                  device='cpu',
                                  state_load_func=custom_state_load_func)

        assert loaded_model.params == model.params
        assert check_weights(model, loaded_model)
        assert isinstance(loaded_model.loss, model.loss.__class__)
        assert isinstance(loaded_model.optimizer, model.optimizer.__class__)
        assert isinstance(loaded_model.prediction_transform,
                          model.prediction_transform.__class__)

    def test_change_state_dict_func(self, saved_argus_model):
        optimizer_state, path, model = saved_argus_model

        def change_state_dict_func(nn_state_dict, optimizer_state_dict):
            nn_state_dict['fc.weight'][0][0] = 0
            if optimizer_state:
                optimizer_state_dict['param_groups'][0]['lr'] = 0.123
            return nn_state_dict, optimizer_state_dict

        loaded_model = load_model(path, change_state_dict_func=change_state_dict_func)
        assert loaded_model.nn_module.state_dict()['fc.weight'][0][0] == 0
        assert model.nn_module.state_dict()['fc.weight'][0][0] != 0
        if optimizer_state:
            assert loaded_model.get_lr() == 0.123
            assert model.get_lr() != 0.123
