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

Solutions of competitions
-------------------------

* `1st place solution for Sensorium Competition at NeurIPS 2023 <https://github.com/lRomul/sensorium>`_
* `1st place solution for SoccerNet Ball Action Spotting Challenge at CVPR 2023 <https://github.com/lRomul/ball-action-spotting>`_
* `1st place solution for SoccerNet Camera Calibration Challenge at CVPR 2023 <https://github.com/NikolasEnt/soccernet-calibration-sportlight>`_
* `1st place solution for Freesound Audio Tagging 2019 at Kaggle <https://github.com/lRomul/argus-freesound>`_
* `14th place solution for TGS Salt Identification Challenge at Kaggle <https://github.com/lRomul/argus-tgs-salt>`_
* `22nd place solution for RANZCR CLiP - Catheter and Line Position Challenge at Kaggle <https://github.com/lRomul/ranzcr-clip>`_
* `45th place solution for RANZCR CLiP - Catheter and Line Position Challenge at Kaggle <https://github.com/irrmnv/ranzcr-clip>`_
* `50th place solution for Quick, Draw! Doodle Recognition Challenge at Kaggle <https://github.com/lRomul/argus-quick-draw>`_
* `66th place solution for Airbus Ship Detection Challenge at Kaggle <https://github.com/OniroAI/Universal-segmentation-baseline-Kaggle-Airbus-Ship-Detection>`_
* `Community Prize solution for Seismic Facies Identification Challenge at AIcrowd <https://github.com/irrmnv/seismic-facies-identification>`_
* `Solution for Deep Chimpact: Depth Estimation for Wildlife Conservation at DrivenData <https://github.com/sankovalev/deep_chimpact.drivendata>`_
* `Solution for Humpback Whale Identification at Kaggle <https://github.com/lRomul/argus-humpback-whale>`_
* `Solution for VSB Power Line Fault Detection at Kaggle <https://github.com/lRomul/argus-vsb-power>`_
* `Solution for Bengali.AI Handwritten Grapheme Classification at Kaggle <https://github.com/lRomul/argus-bengali-ai>`_
* `Solution for ALASKA2 Image Steganalysis competition at Kaggle <https://github.com/lRomul/argus-alaska>`_
