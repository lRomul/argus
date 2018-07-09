
from argus.model.model import Model

TRAIN_ATTRS = {'nn_module', 'optimizer', 'loss', 'device'}
PREDICT_ATTRS = {'nn_module', 'predict_transform', 'device'}
ALL_ATTRS = TRAIN_ATTRS | PREDICT_ATTRS
