import sys
sys.path.append('../')

import numpy as np
import caffe
import os
import re

from lib import run_net
from lib import score_util
from lib import util
from datasets.davis import davis

from PIL import Image
import scipy.io as sio
import matplotlib.pyplot as plt
import cPickle
import time

caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

video_name = sys.argv[2]

net = caffe.Net('../nets/stage-voc-fcn8s.prototxt',
                '../nets/fcn8s-heavy-pascal.caffemodel',
                caffe.TEST)

DV = davis('../data/DAVIS')

with open('../cache/imagesets/val_DAVIS.txt', 'r') as f:
    val = f.read().splitlines()
f.close()

def get_features_fc7(video_name):
    print 'Calculating feature vectors of {}...'.format(video_name)
    num_images = 0
    feature_mean = []
    feature_max = []
    info = []
    for idx in val:
        info_ = idx.split()
        if info_[0] != video_name:
            continue
        #print info_[0], info_[1]
        im = np.array(Image.open('../data/DAVIS/JPEGImages/480p/{}/{:0>5d}.jpg'.format(info_[0], int(info_[1]))))
        
        #start = time.time()
        scoremap, fc7 = util.segrun_fc7(net, DV.preprocess(im))
        #end = time.time()
        #print end - start
        size1 = scoremap.shape
        size2 = fc7.shape
        
        # label segments
        scoremap[scoremap > 0] = 1
        
        if np.sum(scoremap) < size1[0] * size1[1] / 100:
            continue
        
        #print 'Resizing scoremap to', size2[1:3], '...'
        scoremap_tmp = util.resize_output(scoremap, size2)
        
        if np.sum(scoremap_tmp) == 0:
            continue
        
        # save image
        im = Image.fromarray((scoremap*255).astype(np.uint8), mode='P')
        im_dir = '../data/DAVIS/Predicted/480p/{}'.format(info_[0])
        if not os.path.exists(im_dir):
            os.makedirs(im_dir)
        im.save('{}/{:0>5d}.png'.format(im_dir, int(info_[1])))
        
        feature_mean.append(np.nanmean(fc7[:, np.nonzero(scoremap_tmp)[0], np.nonzero(scoremap_tmp)[1]], axis=1))
        feature_max.append(np.nanmax(fc7[:, np.nonzero(scoremap_tmp)[0], np.nonzero(scoremap_tmp)[1]], axis=1))
        info.append([info_[0], info_[1]])
        num_images += 1
    #print num_images
    
    # save the data
    feat_dir = '../cache/features'
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)
    f = open(feat_dir + '/features_DAVIS_' + video_name + '_fc7.p', 'w')
    cPickle.dump({'info': info, 'feature_mean': np.asarray(feature_mean), 'feature_max': np.asarray(feature_max)}, f)
    f.close()

get_features_fc7(video_name)
