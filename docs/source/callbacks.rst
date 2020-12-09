argus.callbacks
===============

.. currentmodule:: argus.callbacks

Callbacks
---------

.. toctree::
   :maxdepth: 2

   ./callbacks/checkpoints
   ./callbacks/early_stopping
   ./callbacks/lr_schedulers
   ./callbacks/logging

Base callback
-------------

.. autoclass:: Callback
   :members:

Decorator callbacks
-------------------

.. autofunction:: on_event

.. autofunction:: on_start

.. autofunction:: on_complete

.. autofunction:: on_epoch_start

.. autofunction:: on_epoch_complete

.. autofunction:: on_iteration_start

.. autofunction:: on_iteration_complete

.. autofunction:: on_catch_exception

.. autoclass:: FunctionCallback
   :members:
