Examples
========

You can find examples `here <https://github.com/lRomul/argus/blob/master/examples>`_.

Basic examples
--------------

* `Quick start. <https://github.com/lRomul/argus/blob/master/examples/quickstart.ipynb>`_
* `MNIST example. <https://github.com/lRomul/argus/blob/master/examples/mnist.py>`_
* `MNIST VAE example. <https://github.com/lRomul/argus/blob/master/examples/mnist_vae.py>`_
* `CIFAR example. <https://github.com/lRomul/argus/blob/master/examples/cifar_simple.py>`_
* `Model loading. <https://github.com/lRomul/argus/blob/master/examples/load_model.py>`_
* `Sequential LR scheduler. <https://github.com/lRomul/argus/blob/master/examples/sequential_lr_scheduler.py>`_

Advanced examples
-----------------

* `CIFAR with DPP, mixed precision and gradient accumulation. <https://github.com/lRomul/argus/blob/master/examples/cifar_advanced.py>`_

    Single GPU training:

    .. code:: bash

        python cifar_advanced.py --batch_size 256 --lr 0.001

    Single machine 2 GPUs distributed data parallel training:

    .. code:: bash

        ./cifar_advanced.sh 2 --batch_size 128 --lr 0.0005

    DDP training with mixed precision and gradient accumulation:

    .. code:: bash

        ./cifar_advanced.sh 2 --batch_size 128 --lr 0.0005 --amp --iter_size 2

* `Custom callback events. <https://github.com/lRomul/argus/blob/master/examples/custom_events.py>`_
* `Custom build methods for creation of model parts. <https://github.com/lRomul/argus/blob/master/examples/custom_build_methods.py>`_

Kaggle solutions
----------------

* `1st place solution for Freesound Audio Tagging 2019 (mel-spectrograms, mixed precision) <https://github.com/lRomul/argus-freesound>`_
* `14th place solution for TGS Salt Identification Challenge (segmentation, MeanTeacher) <https://github.com/lRomul/argus-tgs-salt>`_
* `22nd place solution for RANZCR CLiP - Catheter and Line Position Challenge (DDP, EMA, mixed precision, pseudo labels) <https://github.com/lRomul/ranzcr-clip>`_
* `50th place solution for Quick, Draw! Doodle Recognition Challenge (gradient accumulation, training on 50M images) <https://github.com/lRomul/argus-quick-draw>`_
* `66th place solution for Kaggle Airbus Ship Detection Challenge (instance segmentation) <https://github.com/OniroAI/Universal-segmentation-baseline-Kaggle-Airbus-Ship-Detection>`_
* `Solution for Humpback Whale Identification (metric learning: arcface, center loss) <https://github.com/lRomul/argus-humpback-whale>`_
* `Solution for VSB Power Line Fault Detection (1d conv) <https://github.com/lRomul/argus-vsb-power>`_
* `Solution for Bengali.AI Handwritten Grapheme Classification (EMA, mixed precision, CutMix) <https://github.com/lRomul/argus-bengali-ai>`_
* `Solution for ALASKA2 Image Steganalysis competition (DDP, EMA, mixed precision, BitMix) <https://github.com/lRomul/argus-alaska>`_
