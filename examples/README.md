# Examples

## Basic examples

* Quick start [quickstart.ipynb](quickstart.ipynb).
* MNIST example [mnist.py](mnist.py).    
* MNIST VAE example [mnist_vae.py](mnist_vae.py).
* CIFAR example [cifar_simple.py](cifar_simple.py). 
* Model loading [load_model.py](load_model.py).
* Sequential LR scheduler [sequential_lr_scheduler.py](sequential_lr_scheduler.py).

## Advanced examples

* CIFAR with DDP, mixed precision and gradient accumulation [cifar_advanced.py](cifar_advanced.py).

    Single GPU training:

    ```bash
    python cifar_advanced.py --batch_size 256 --lr 0.001
    ```
 
    Single machine 2 GPUs distributed data parallel training:

    ```bash
    ./cifar_advanced.sh 2 --batch_size 128 --lr 0.0005
    ```

    DDP training with mixed precision and gradient accumulation:

    ```bash
    ./cifar_advanced.sh 2 --batch_size 128 --lr 0.0005 --amp --iter_size 2
    ```
    
* Custom callback events [custom_events.py](custom_events.py).
* Custom build methods for creation of model parts [custom_build_methods.py](custom_build_methods.py).

## Solutions of competitions

* [1st place solution for Sensorium Competition at NeurIPS 2023](https://github.com/lRomul/sensorium)
* [1st place solution for SoccerNet Ball Action Spotting Challenge at CVPR 2023](https://github.com/lRomul/ball-action-spotting)
* [1st place solution for SoccerNet Camera Calibration Challenge at CVPR 2023](https://github.com/NikolasEnt/soccernet-calibration-sportlight)
* [1st place solution for Freesound Audio Tagging 2019 at Kaggle](https://github.com/lRomul/argus-freesound)
* [14th place solution for TGS Salt Identification Challenge at Kaggle](https://github.com/lRomul/argus-tgs-salt)
* [22nd place solution for RANZCR CLiP - Catheter and Line Position Challenge at Kaggle](https://github.com/lRomul/ranzcr-clip)
* [45th place solution for RANZCR CLiP - Catheter and Line Position Challenge at Kaggle](https://github.com/irrmnv/ranzcr-clip)
* [50th place solution for Quick, Draw! Doodle Recognition Challenge at Kaggle](https://github.com/lRomul/argus-quick-draw)
* [66th place solution for Airbus Ship Detection Challenge at Kaggle](https://github.com/OniroAI/Universal-segmentation-baseline-Kaggle-Airbus-Ship-Detection)
* [Community Prize solution for Seismic Facies Identification Challenge at AIcrowd](https://github.com/irrmnv/seismic-facies-identification)
* [Solution for Deep Chimpact: Depth Estimation for Wildlife Conservation at DrivenData](https://github.com/sankovalev/deep_chimpact.drivendata)
* [Solution for Humpback Whale Identification at Kaggle](https://github.com/lRomul/argus-humpback-whale)
* [Solution for VSB Power Line Fault Detection at Kaggle](https://github.com/lRomul/argus-vsb-power)
* [Solution for Bengali.AI Handwritten Grapheme Classification at Kaggle](https://github.com/lRomul/argus-bengali-ai)
* [Solution for ALASKA2 Image Steganalysis competition at Kaggle](https://github.com/lRomul/argus-alaska)
