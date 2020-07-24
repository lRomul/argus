import torch

from torchvision.models import densenet121

from argus import Model
from argus.model.build import (
    choose_attribute_from_dict,
    cast_optimizer,
    cast_nn_module
)


class MyModel(Model):
    nn_module = densenet121
    optimizer = torch.optim.Adam
    loss = torch.nn.CrossEntropyLoss

    def build_nn_module(self, nn_module_meta, nn_module_params):
        if nn_module_meta is None:
            raise ValueError("nn_module is required attribute for argus.Model")

        nn_module, nn_module_params = choose_attribute_from_dict(nn_module_meta,
                                                                 nn_module_params)
        nn_module = cast_nn_module(nn_module)
        nn_module = nn_module(**nn_module_params)

        # Replace last fully connected layer
        num_classes = self.params['num_classes']
        in_features = nn_module.classifier.in_features
        nn_module.classifier = torch.nn.Linear(in_features=in_features,
                                               out_features=num_classes)
        return nn_module

    def build_optimizer(self, optimizer_meta, optim_params):
        optimizer, optim_params = choose_attribute_from_dict(optimizer_meta,
                                                             optim_params)
        optimizer = cast_optimizer(optimizer)

        # Set small LR for pretrained layers
        pretrain_modules = [
            self.nn_module.features
        ]
        pretrain_params = []
        for pretrain_module in pretrain_modules:
            pretrain_params += pretrain_module.parameters()
        pretrain_ids = list(map(id, pretrain_params))
        other_params = filter(lambda p: id(p) not in pretrain_ids,
                              self.nn_module.parameters())
        grad_params = [
            {"params": pretrain_params, "lr": optim_params['lr'] * 0.01},
            {"params": other_params, "lr": optim_params['lr']}
        ]
        del optim_params['lr']

        optimizer = optimizer(params=grad_params, **optim_params)
        return optimizer


if __name__ == "__main__":
    params = {
        'nn_module': {'pretrained': True, 'progress': False},
        'optimizer': {'lr': 0.001},
        'device': 'cuda',
        'num_classes': 10
    }

    model = MyModel(params)
    print("Learning rate for each params group:", model.get_lr())
    print("Last FC layer:", model.nn_module.classifier)
