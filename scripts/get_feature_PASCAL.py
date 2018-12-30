import sys
sys.path.append('../')

import numpy as np
import caffe
import os
import cPickle

from lib import run_net
from lib import score_util
from lib import util
from datasets.pascal_voc import pascal

from scipy.ndimage import label, generate_binary_structure
from PIL import Image

caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

net = caffe.Net('../nets/stage-voc-fcn8s.prototxt',
                '../nets/fcn8s-heavy-pascal.caffemodel',
                caffe.TEST)

PV = pascal('../data/PASCAL/VOC2011')

pv_classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
              'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person',
              'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def load_dataset(class_name):
    dir_path = '../data/PASCAL/VOC2011'
    with open('{}/ImageSets/Main/{}_train.txt'.format(dir_path, class_name), 'r') as f:
        inputs = f.read().splitlines()
    f.close()
    return inputs

def get_features_fc7(class_name):
    print 'Calculating feature vectors of {}...'.format(class_name)
    inputs = load_dataset(class_name)
    num_images = 0
    feature_mean = []
    feature_max = []
    info = []
    for idx in inputs:
        info_ = idx.split()
        if info_[1] == '-1':
            continue
        #print class_name, info_[0]
        im = PV.load_image(info_[0])
        scoremap, fc7 = util.segrun_fc7(net, PV.preprocess(im))
        size1 = scoremap.shape
        size2 = fc7.shape
        
        # label segments
        scoremap[scoremap != pv_classes.index(class_name)] = 0
        scoremap[scoremap > 0] = 1
        
        if np.sum(scoremap) < size1[0] * size1[1] / 100:
            continue
        
        #print 'Resizing scoremap to', size2[1:3], '...'
        scoremap_tmp = util.resize_output(scoremap, size2)
        
        if np.sum(scoremap_tmp) == 0:
            continue
        
        # save images
        im = Image.fromarray(scoremap * 255)
        im_dir = '../data/PASCAL/VOC2011/Predicted/{}'.format(class_name)
        if not os.path.exists(im_dir):
            os.makedirs(im_dir)
        im.save('{}/{}.png'.format(im_dir, info_[0]))
        
        feature_mean.append(np.nanmean(fc7[:, np.nonzero(scoremap_tmp)[0], np.nonzero(scoremap_tmp)[1]], axis=1))
        feature_max.append(np.nanmax(fc7[:, np.nonzero(scoremap_tmp)[0], np.nonzero(scoremap_tmp)[1]], axis=1))
        info.append([class_name, info_[0]])
        num_images += 1
    #print num_images
    
    # save the data
    feat_dir = '../cache/features'
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)
    f = open(feat_dir + '/features/features_PASCAL_' + class_name + '_fc7.p', 'w')
    cPickle.dump({'info': info, 'feature_mean': np.asarray(feature_mean), 'feature_max': np.asarray(feature_max)}, f)
    f.close()

for pv_class in pv_classes[1:len(pv_classes)]:
    get_features_fc7(pv_class)
