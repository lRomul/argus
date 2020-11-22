import math
import torch
import warnings

from argus.callbacks import Callback
from argus.engine import State


METRIC_REGISTRY = {}


def init_better(better: str, monitor: str) -> tuple:
    if better not in ['min', 'max', 'auto']:
        raise ValueError(f"Unknown better option '{better}'")

    if better == 'auto':
        if monitor.startswith('val_'):
            metric_name = monitor[len('val_'):]
        else:
            metric_name = monitor[len('train_'):]
        if metric_name not in METRIC_REGISTRY:
            raise ImportError(f"Metric '{metric_name}' not found in scope")
        better = METRIC_REGISTRY[metric_name].better

    if better == 'min':
        better_comp = lambda a, b: a < b
        best_value = math.inf
    else:  # better == 'max':
        better_comp = lambda a, b: a > b
        best_value = -math.inf

    return better, better_comp, best_value


class MetricMeta(type):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        new_class = super().__new__(mcs, name, bases, attrs)
        metric_name = attrs['name']
        if metric_name:
            if metric_name in METRIC_REGISTRY:
                current_class = f"<class '{attrs['__module__']}.{attrs['__qualname__']}'>"
                warnings.warn(f"{current_class} redefined '{metric_name}' "
                              f"that was already registered by {METRIC_REGISTRY[metric_name]}")
            METRIC_REGISTRY[metric_name] = new_class
        return new_class


class Metric(Callback, metaclass=MetricMeta):
    """Base metric class.

    To create a custom metric needs to create a class inheriting from Metric:

    * Override three methods: reset, update, and compute.
    * Set class attribute: name, better.

    Attributes:
        name (str): Metric name. Defaults to ''.
        better (str): Minimization or maximization is better. Should be ‘min’
            or ‘max’. It will be used, for example, by
            :class:`argus.callbacks.MonitorCheckpoint`. Defaults to 'min'.

    Example:

        MAP@k implementation:

        .. code-block:: python

            import torch
            import numpy as np
            from argus.metrics import Metric


            def apk(actual, predicted, k=3):
                if len(predicted) > k:
                    predicted = predicted[:k]

                score = 0.0
                num_hits = 0.0

                for i, p in enumerate(predicted):
                    if p in actual and p not in predicted[:i]:
                        num_hits += 1.0
                        score += num_hits / (i+1.0)

                if not actual:
                    return 0.0

                return score / min(len(actual), k)


            def mapk(actual, predicted, k=3):
                return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


            class MAPatK(Metric):
                name = 'map_at_k'
                better = 'max'

                def __init__(self, k=3):
                    super().__init__()
                    self.k = k
                    self.scores = []

                def reset(self):
                    self.scores = []

                def update(self, step_output: dict):
                    preds = step_output['prediction'].cpu().numpy()
                    trgs = step_output['target'].cpu().numpy()

                    preds_idx = preds.argsort(axis=1)
                    preds_idx = np.fliplr(preds_idx)[:, :self.k]

                    self.scores += [apk([a], p, self.k) for a, p in zip(trgs, preds_idx)]

                def compute(self):
                    return np.mean(self.scores)

        Then you can use the metric like this:

        .. code-block:: python

            callbacks = [
                MonitorCheckpoint(dir_path='mnist', monitor='val_map_at_k', better='auto')
            ]

            model.fit(train_loader,
                      val_loader=val_loader,
                      metrics=['map_at_k'],  # or the same: metrics=[MAPatK(k=3)]
                      callbacks=callbacks)

    """

    name: str = ''
    better: str = 'min'

    def reset(self):
        """Init or resets internal variables and accumulators."""
        pass

    def update(self, step_output: dict):
        """Updates internal variables with provided *step_output*.

        *step_output* from default :meth:`argus.model.Model.train_step` and
        :meth:`argus.model.Model.val_step` looks like::

            {
                'prediction': The batch predictions,
                'target': The batch targets,
                'loss': Loss function value
            }

        """
        pass

    def compute(self):
        """Computes custom metric and return the result."""
        pass

    def epoch_start(self, state: State):
        self.reset()

    def iteration_complete(self, state: State):
        self.update(state.step_output)

    def epoch_complete(self, state: State):
        """Stores metric value to :class:`argus.engine.State`.
        You can override this method if you want, for example, to save several
        metrics values in the state.
        """
        with torch.no_grad():
            score = self.compute()
        name_prefix = f"{state.phase}_" if state.phase else ''
        state.metrics[name_prefix + self.name] = score
