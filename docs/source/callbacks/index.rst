argus.callbacks
===============

All callbacks classes should inherit the base :class:`~argus.callbacks.Callback` class.

A callback may execute actions on the start and the end of the whole training
process, each epoch or iteration, as well as any other custom events. The
actions should be specified within corresponding functions:
*event*, *start*, *complete*, *epoch_start*, *epoch_complete*,
*iteration_start*, *iteration_complete*.

A simple custom callback which stops training after the specified time:

.. code-block:: python

    from time import time

    from argus.engine import State
    from argus.callbacks.callback import Callback


    class TimerCallback(Callback):
        def __init__(self, time_limit: int):
            self.time_limit = time_limit
            self.start_time = 0

        def epoch_start(self, state: State):
            if state.epoch == 0:
                 self.start_time = time()

        def iteration_complete(self, state: State):
            if time() - self.start_time > self.time_limit:
                state.stopped = True
                state.logger.info("Run out of time!")


.. currentmodule:: argus.callbacks

.. autoclass:: Callback
   :members:


.. toctree::
   :maxdepth: 2
   :caption: Сallbacks

   ./checkpoints
   ./early_stopping
   ./logging
