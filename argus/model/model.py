from argus.model.build import BuildModel


class Model(BuildModel):
    def __init__(self, params):
        super().__init__(params)

    def _train_step(self):
        pass

    def _val_step(self):
        pass

    def fit(self, train_loader, val_loader=None, max_epochs=None, handlers=None):
        pass

    def set_lr(self, lr):
        pass

    def save(self, file_path):
        pass

    def validate(self, val_loader):
        pass

    def predict(self, input_):
        pass
