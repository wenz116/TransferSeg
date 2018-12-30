from __future__ import division
import caffe
import numpy as np
import os
import sys
import re
from datetime import datetime
from PIL import Image

def compute_iou(net, save_dir, dataset, layer='score_weight', gt='label'):
    iou_all = {}
    iou_sum = 0
    loss = 0
    shot_num = 0
    for idx in dataset:
        net.forward()
        
        info = idx.split()
        label_true = np.array(Image.open('../data/DAVIS/Annotations/480p/{}/{:0>5d}.png'.format(info[0], int(info[1]))))
        label_true[label_true > 0] = 1
        label_pred = net.blobs[layer].data[0]
        
        in_ = np.zeros(label_pred.shape[1:])
        fgbg_map = np.zeros(label_pred.shape[1:])
        
        fgbg_map[label_pred[0,:,:] > (np.mean(label_pred) * 2 + np.max(label_pred)) / 3] = 1
        iou = np.sum(label_true * fgbg_map) / np.sum(label_true + fgbg_map - label_true * fgbg_map)
        iou_sum += iou
        
        if info[0] not in iou_all:
            iou_all[info[0]] = {}
        iou_all[info[0]][info[1]] = iou
        
        if save_dir:
            im = Image.fromarray((fgbg_map*255).astype(np.uint8), mode='P')
            if save_dir == 1:
                im_dir = '../data/DAVIS/Predicted_1/480p/{}'.format(info[0])
            else:
                im_dir = '../data/DAVIS/Predicted_2/480p/{}'.format(info[0])
            if not os.path.exists(im_dir):
                os.makedirs(im_dir)
            im.save('{}/{:0>5d}.png'.format(im_dir, int(info[1])))
        
        # compute the loss as well
        loss += net.blobs['loss'].data.flat[0]
        
        shot_num += 1

    #print iou_sum / shot_num, shot_num
    return iou_sum / shot_num, iou_all, loss / len(dataset)

def seg_tests(solver, save_format, dataset, layer='score_weight', gt='label'):
    print '>>>', datetime.now(), 'Begin seg tests'
    solver.test_nets[0].share_with(solver.net)
    iou, iou_all = do_seg_tests(solver.test_nets[0], solver.iter, save_format, dataset, layer, gt)
    return iou, iou_all

def do_seg_tests(net, iter, save_format, dataset, layer='score_weight', gt='label'):
    n_cl = net.blobs[layer].channels
    iou, iou_all, loss = compute_iou(net, save_format, dataset, layer, gt)
    print '>>>', datetime.now(), 'Iteration', iter, 'loss', loss
    print '>>>', datetime.now(), 'Iteration', iter, 'iou', iou
    return iou, iou_all
