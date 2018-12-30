import os
import glob
import numpy as np
import PIL
from PIL import Image


class davis:
    def __init__(self, data_path):
        self.dir = data_path
        # same mean as PASCAL VOC (the imagenet mean) for compatibility w/ FCN
        self.mean = (104.00698793, 116.66876762, 122.67891434)
        self.MAX_DIM = 500.0  # match PASCAL VOC training data
        self.label_thresh = 15

    def resize(self, im, label=False):
        dims = np.array(im).shape
        if len(dims) > 2:
            dims = dims[:-1]
        max_val, max_idx = np.max(dims), np.argmax(dims)
        scale = self.MAX_DIM / max_val
        new_height, new_width = int(dims[0]*scale), int(dims[1]*scale)
        if label:
            im = im.resize((new_width, new_height), resample=PIL.Image.NEAREST)
        else:
            im = im.resize((new_width, new_height), resample=PIL.Image.BILINEAR)
        return im

    def preprocess(self, im):
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:, :, ::-1]
        in_ -= np.array(self.mean)
        in_ = in_.transpose((2, 0, 1))
        return in_
