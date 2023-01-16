# DCSRN
## Introduction
This project is based on replication of article: [BRAIN MRI SUPER RESOLUTION USING 3D DEEP DENSELY CONNECTED NEURAL NETWORKS](https://arxiv.org/abs/1801.02728), which utilizes deep networks to perform super resolution on 3D models from CT/MRI scans.

## Data Format Requirements
data organization:
data:
|
|-----3d_hr_data.npy
|-----3d_lr_data.npy

Both should have the numpy shape (N, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1) where N is the number of patches available.
default: (N, 64, 64, 64, 1)

## Training
Use the following command to check possible input arguements for training.
```
python dcsrn.py -h 
```
