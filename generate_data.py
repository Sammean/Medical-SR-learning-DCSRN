import os
from glob import glob
import numpy as np


def get_data(root):
    paths = sorted(glob(os.path.join(root, "*_*.npy")))
    data = np.array([np.load(fname) for fname in paths], dtype='float32')
    return data


def integrate_data():
    lr_root = './data/LD/'
    hr_root = './data/HD/'
    lr_data = get_data(lr_root)
    hr_data = get_data(hr_root)
    print(lr_data.shape, hr_data.shape)
    np.save('./data/3d_lr_data2', lr_data)
    np.save('./data/3d_hr_data2', hr_data)
    print('Convert to .npy Done!!!')


def tow2one(f1, f2):
    d1 = np.array(np.load(f1), dtype='float32')
    d2 = np.array(np.load(f2), dtype='float32')
    d = np.concatenate((d1,d2), axis=0)
    d = d.reshape((-1, 64, 64, 64, 1))
    print(d.shape)
    # np.save('./data/3d_hr_data', d)
    np.save('./data/3d_hr_data', d)
    print('concat done!!!')


if __name__ == '__main__':
    # root = 'E:/BUAA/Project/02-SR/IXI/IXI_HR/T1/train/'
    # hr_data = get_hr_data(root)
    # lr_data = np.array([get_LR(hd) for hd in hr_data], dtype='float32')
    # print('Generate LR data Done!!!')
    # print(hr_data.shape, lr_data.shape)
    # np.save('./data/3d_lr_data', lr_data)
    # np.save('./data/3d_hr_data', hr_data)
    # print('Convert to .npy Done!!!')

    # integrate_data()

    # l1, l2 = './data/3d_lr_data1.npy', './data/3d_lr_data2.npy'
    # tow2one(l1, l2)
    h1, h2 = './data/3d_hr_data1.npy', './data/3d_hr_data2.npy'
    tow2one(h1, h2)





