import os.path
import time
import numpy as np


def random_patching(lr_data, hr_data, patch_size=64):
    np.random.seed(int(time.time()))
    x_shape = lr_data.shape[0]
    y_shape = lr_data.shape[1]
    z_shape = lr_data.shape[2]
    x_idx = np.random.randint(0, x_shape-patch_size)
    y_idx = np.random.randint(0, y_shape-patch_size)
    z_idx = np.random.randint(0, z_shape-patch_size)
    lr_patch, hr_patch = lr_data[x_idx:x_idx+patch_size,y_idx:y_idx+patch_size,z_idx:z_idx+patch_size]\
        , hr_data[x_idx:x_idx+patch_size,y_idx:y_idx+patch_size,z_idx:z_idx+patch_size]
    return lr_patch, hr_patch

def get_patching(root, patch_size=64):

    lr_path = os.path.join(root,'3d_lr_data_o.npy')
    hr_path = os.path.join(root,'3d_hr_data_o.npy')
    lr_data = np.array(np.load(lr_path), dtype='float32')
    hr_data = np.array(np.load(hr_path), dtype='float32')
    print(lr_data.shape, hr_data.shape)
    lr_patches = []
    hr_patches = []
    for i in range(lr_data.shape[0]):
        lp, hp = random_patching(lr_data[i], hr_data[i], patch_size)
        lr_patches.append(lp)
        hr_patches.append(hp)
    lr_patches, hr_patches = np.array(lr_patches, dtype='float32'), np.array(hr_patches, dtype='float32')
    lr_patches, hr_patches = lr_patches.reshape((-1, patch_size, patch_size, patch_size, 1)), hr_patches.reshape((-1, patch_size, patch_size, patch_size, 1))
    print(lr_patches.shape, hr_patches.shape)

    np.save(os.path.join(root,'3d_lr_data1.npy'), lr_patches)
    np.save(os.path.join(root,'3d_hr_data1.npy'), hr_patches)
    print('random patching done!!!')

if __name__ == '__main__':
    root = '../DCSRN-main/data/'
    get_patching(root)







