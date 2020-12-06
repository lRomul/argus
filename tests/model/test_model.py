import torch
from torch.utils.data import TensorDataset, DataLoader

from argus import load_model
from argus.engine import State


class TestModelMethod:
    def test_train_step(self, linear_argus_model_instance, poly_batch):
        model = linear_argus_model_instance
        batch_size = poly_batch[0].shape[0]
        output = model.train_step(
            poly_batch,
            State(linear_argus_model_instance.test_step)
        )

        assert isinstance(output, dict)
        prediction = output['prediction']
        target = output['target']
        loss = output['loss']
        assert isinstance(prediction, torch.Tensor)
        assert list(prediction.shape) == [batch_size, 1]
        assert isinstance(target, torch.Tensor)
        assert list(target.shape) == [batch_size, 1]
        assert isinstance(loss, float)

    def test_val_step(self, linear_argus_model_instance, poly_batch):
        model = linear_argus_model_instance
        batch_size = poly_batch[0].shape[0]
        output = model.val_step(
            poly_batch,
            State(linear_argus_model_instance.test_step)
        )

        assert isinstance(output, dict)
        prediction, target = output['prediction'], output['target']
        assert isinstance(prediction, torch.Tensor)
        assert list(prediction.shape) == [batch_size, 1]
        assert isinstance(target, torch.Tensor)
        assert list(target.shape) == [batch_size, 1]

    def test_predict(self, linear_argus_model_instance, poly_batch):
        model = linear_argus_model_instance

        predict = model.predict(poly_batch[0])
        assert isinstance(predict, torch.Tensor)
        assert list(predict.shape) == [poly_batch[0].shape[0], 1]

    def test_train_val_modes(self, linear_argus_model_instance):
        model = linear_argus_model_instance
        model.eval()
        assert not model.nn_module.training
        model.train()
        assert model.nn_module.training

    def test_fit_train_loader(self,
                              get_batch_function,
                              linear_argus_model_instance):
        model = linear_argus_model_instance
        train_dataset = TensorDataset(*get_batch_function(batch_size=1024))
        train_loader = DataLoader(train_dataset, shuffle=True,
                                  drop_last=True, batch_size=32)
        val_loss_before = model.validate(train_loader)['val_loss']
        model.fit(train_loader, num_epochs=4)
        val_loss_after = model.validate(train_loader)['val_loss']
        assert val_loss_after < val_loss_before

    def test_fit_train_val_loader(self,
                                  get_batch_function,
                                  linear_argus_model_instance):
        model = linear_argus_model_instance
        train_dataset = TensorDataset(*get_batch_function(batch_size=4096))
        val_dataset = TensorDataset(*get_batch_function(batch_size=512))
        train_loader = DataLoader(train_dataset, shuffle=True,
                                  drop_last=True, batch_size=32)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=64)
        val_loss_before = model.validate(val_loader)['val_loss']
        model.fit(train_loader,
                  val_loader=val_loader,
                  num_epochs=32)
        val_loss_after = model.validate(val_loader)['val_loss']
        assert val_loss_after < val_loss_before
        assert val_loss_after < 0.3

    def test_save(self, tmpdir, linear_argus_model_instance):
        path = str(tmpdir.mkdir("experiment").join("model.pth"))
        linear_argus_model_instance.save(path)
        model = load_model(path, device='cpu')
        assert model.params == linear_argus_model_instance.params

        state = torch.load(path)
        assert set(state.keys()) == {'model_name', 'params', 'nn_state_dict'}

    def test_save_with_optimizer_state(self, tmpdir, linear_argus_model_instance):
        path = str(tmpdir.mkdir("experiment").join("model.pth"))
        linear_argus_model_instance.save(path, optimizer_state=True)
        model = load_model(path, device='cpu')
        assert model.params == linear_argus_model_instance.params

        state = torch.load(path)
        assert set(state.keys()) == {'model_name', 'params',
                                     'nn_state_dict', 'optimizer_state_dict'}
