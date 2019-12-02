import torch
import numpy as np
from torch.utils.data import Dataset
from utils.preprocess import preprocess_image, get_mask_and_regr, imread, get_overlay_mask
import os.path
from os import path

class baseline(Dataset):
    """Car dataset."""

    def __init__(self, dataframe, root_dir, training=True, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image name
        idx, labels = self.df.values[idx]
        img_name = self.root_dir.format(idx)
        

        # Augmentation
        flip = False
        if self.training:
            flip = np.random.randint(10) == 1
        
        # Read imae
        img0 = imread(img_name, True)

        if path.exists('../data/heatmaps/{}.npy'.format(idx)):
            mask0 = np.load('../data/heatmaps/{}.npy'.format(idx))

        else:
            mask0 = get_overlay_mask(labels, img0)
            np.save('../data/heatmaps/{}.npy'.format(idx), mask0)
        
        img = preprocess_image(img0, flip=flip)
        img = np.rollaxis(img, 2, 0)
        
        mask0 = mask0
        mask0 = preprocess_image(mask0 * 255, flip=flip, mask = True)
        mask0 = mask0[:,:,2]
        
        # Get mask and regression maps
        mask, regr = get_mask_and_regr(img0, mask0, labels, flip=flip)
        regr = np.rollaxis(regr, 2, 0)
        
        return [img, mask, regr, (labels, idx)]

