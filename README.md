# SDUNet-ISBIDataset-2012
Simplified SD-UNet for cell segmentation on ISBI 2012 dataset using PyTorch.

This repository contains a PyTorch implementation of SDUNet (Simplified Depthwise U-Net) for cell segmentation of neuronal structures in electron microscopic on the ISBI 2012 dataset.The model uses depthwise separable convolutions and weight-standardized convolutions to train on low computation devices.

This implementation is based on the SDUNet architecture described in “Stripping down U-Net for Segmentation of Biomedical Images on Platforms with Low Computational Budgets”.

Features:

a)Simplified U-Net architecture for biomedical image segmentation
b)Depthwise separable and weight-standardized convolutions
c)Visualization of input images, ground truth masks, and predicted masks

Environment / Platform:

Developed and tested on Google Colaboration with GPU T4 (free version)
Python 3.x, PyTorch, and common packages (torchvision, numpy, matplotlib, tifffile).
Dataset used: ISBI 2012 (downloaded from Kaggle)

Installation:
git clone https://github.com/GaargiHA/SDUNet-ISBIDataset-2012.git
cd SDUNet-ISBIDataset-2012
pip install -r requirements.txt  # if you have a requirements file

Usage:

Prepare your ISBI 2012 dataset (train volumes and labels)
Set the correct paths in the code
Run the training script - python train_sdunet.py

Reference:

1.P. K. Gadosey et al., “SD UNet: Stripping down U Net for segmentation of biomedical images on platforms with low computational budgets,” Diagnostics, 2020.

2.PyTorch tutorials: Writing Custom Datasets, DataLoaders and Transforms. https://pytorch.org/tutorials/beginner/data_loading_tutorial.html


