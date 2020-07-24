# Examples

## Basic examples

* MNIST example.
    ```bash
    python mnist.py --device cuda
    ```
    
* MNIST VAE example. 
    ```bash
    python mnist_vae.py --device cuda
    ```
  
* CIFAR example. 
    ```bash
    python cifar_simple.py --device cuda
    ```

Advanced examples
-----------------

* Advanced CIFAR with DPP, mixed precision and gradient accumulation.

    Single GPU training:

    ```bash
    python cifar_advanced.py --batch_size 256 --lr 0.001
    ```
 
    Single machine 2 GPUs distributed data parallel training:

    ```bash
    ./cifar_advanced.sh 2 --batch_size 128 --lr 0.0005
    ```

    DDP training with Apex mixed precision and gradient accumulation:

    ```bash
    ./cifar_advanced.sh 2 --batch_size 128 --lr 0.0005 --amp --iter_size 2
    ```
    
* Custom build methods for creation of model parts [custom_build_methods.py](custom_build_methods.py).

Kaggle solutions
----------------

* [1st place solution for Freesound Audio Tagging 2019 (mel-spectrograms, mixed precision)](https://github.com/lRomul/argus-freesound)
* [14th place solution for TGS Salt Identification Challenge (segmentation, MeanTeacher)](https://github.com/lRomul/argus-tgs-salt)
* [50th place solution for Quick, Draw! Doodle Recognition Challenge (gradient accumulation, training on 50M images)](https://github.com/lRomul/argus-quick-draw)
* [66th place solution for Kaggle Airbus Ship Detection Challenge (instance segmentation)](https://github.com/OniroAI/Universal-segmentation-baseline-Kaggle-Airbus-Ship-Detection)
* [Solution for Humpback Whale Identification (metric learning: arcface, center loss)](https://github.com/lRomul/argus-humpback-whale)
* [Solution for VSB Power Line Fault Detection (1d conv)](https://github.com/lRomul/argus-vsb-power)
* [Solution for Bengali.AI Handwritten Grapheme Classification (EMA, mixed precision, CutMix)](https://github.com/lRomul/argus-bengali-ai)
* [Solution for ALASKA2 Image Steganalysis competition (DDP, EMA, mixed precision, BitMix)](https://github.com/lRomul/argus-alaska)
