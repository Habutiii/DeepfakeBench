#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Create environment with Python 3.7
conda create -n deepfakebench python=3.7 -y

# Activate environment (this works in interactive shells; use eval for scripts)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deepfakebench

# Install packages from conda
conda install -y  --override-channels -c conda-forge -c nvidia -c pytorch \
    numpy=1.21.5 pandas=1.3.5 pillow=9.0.1 dlib=19.24.0 imageio=2.9.0 \
    tqdm=4.61.0 scipy=1.7.3 seaborn=0.11.2 pyyaml=6.0 \
    scikit-image=0.19.2 scikit-learn=1.0.2 \
    pytorch torchvision torchaudio pytorch-cuda \
    setuptools simplejson einops filterpy

# Install pip-only packages
pip install opencv-python==4.6.0.66 einops==0.4.1 \
    lmdb imgaug==0.4.0 imutils==0.5.4 albumentations==1.1.0 \
    efficientnet-pytorch==0.7.1 timm==0.6.12 segmentation-models-pytorch==0.3.2 \
    torchtoolbox==0.1.8.2 tensorboard==2.10.1 loralib pytorchvideo \
    kornia transformers==4.26.1 fvcore git+https://github.com/openai/CLIP.git

echo "Environment 'deepfakebench' setup complete."
