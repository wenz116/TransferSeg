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
from scipy.ndimage import label, generate_binary_structure
from sklearn.metrics.pairwise import cosine_similarity
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

def get_similarity(video_name):
    with open('../cache/features/features_DAVIS_' + video_name + '_fc7.p', 'r') as f:
        feat_dict = cPickle.load(f)
    f.close()
    
    info = feat_dict['info']
    feature_mean = feat_dict['feature_mean']
    feature_max = feat_dict['feature_max']
    
    list_cos_sim = []
    
    pv_classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                  'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    
    for pv_class in pv_classes[1:len(pv_classes)]:
        with open('../cache/features/features_PASCAL_' + pv_class + '_fc7.p', 'r') as f:
            pv_feat_dict = cPickle.load(f)
        f.close()
        
        pv_info = pv_feat_dict['info']
        pv_feature_mean = pv_feat_dict['feature_mean']
        pv_feature_max = pv_feat_dict['feature_max']
        
        cos_sim = np.zeros((len(info), len(pv_info)))
        for i in range(len(info)):
            vec_1 = feature_mean[i].reshape(1,-1)
            for j in range(len(pv_info)):
                vec_2 = pv_feature_mean[j].reshape(1,-1)
                cos_sim[i, j] = cosine_similarity(vec_1, vec_2)
        list_cos_sim.append(np.nanmean(np.nanmax(cos_sim, axis=1)))
        #print video_name, pv_class, np.nanmean(np.nanmax(cos_sim, axis=1))
    return list_cos_sim

def get_features(video_name):  # feature similarity
    num_images = 0
    num_segments = 0
    feature_vectors = []
    score_vectors = []
    motion_vectors = []
    info_seg = []
    masks = []
    first = 1
    
    with open('../cache/features/features_DAVIS_' + video_name + '_fc7.p', 'r') as f:
        feat_dict = cPickle.load(f)
    f.close()
    
    info = feat_dict['info']
    feature_mean = feat_dict['feature_mean']
    feature_max = feat_dict['feature_max']
    
    list_cos_sim = []
    
    pv_classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                  'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    
    for pv_class in pv_classes[1:len(pv_classes)]:
        with open('../cache/features/features_PASCAL_' + pv_class + '_fc7.p', 'r') as f:
            pv_feat_dict = cPickle.load(f)
        f.close()
        
        pv_info = pv_feat_dict['info']
        pv_feature_mean = pv_feat_dict['feature_mean']
        pv_feature_max = pv_feat_dict['feature_max']
        
        cos_sim = np.zeros((len(info), len(pv_info)))
        for i in range(len(info)):
            vec_1 = feature_mean[i].reshape(1,-1)
            for j in range(len(pv_info)):
                vec_2 = pv_feature_mean[j].reshape(1,-1)
                cos_sim[i, j] = cosine_similarity(vec_1, vec_2)
        list_cos_sim.append(np.nanmean(np.nanmax(cos_sim, axis=1)))
        #print video_name, pv_class, np.nanmean(np.nanmax(cos_sim, axis=1))
    
    list_index = range(1, 21)
    list_cos_sim, list_index = zip(*sorted(zip(list_cos_sim, list_index), reverse=True))
    
    for idx in val:
        info_ = idx.split()
        if info_[0] != video_name:
            continue
        im = np.array(Image.open('../data/DAVIS/JPEGImages/480p/{}/{:0>5d}.jpg'.format(info_[0], int(info_[1]))))
        
        #start = time.time()
        scoremap, score, conv5_3, conv4_3, conv3_3, conv2_2, conv1_2 = util.segrun(net, DV.preprocess(im))
        #end = time.time()
        #print end - start
        
        size = score.shape
        score = np.asarray([score])
        """
        print 'Resizing feature maps to', size, '...'
        conv5_3 = util.resize_output(conv5_3, size)
        conv4_3 = util.resize_output(conv4_3, size)
        conv3_3 = util.resize_output(conv3_3, size)
        conv2_2 = util.resize_output(conv2_2, size)
        conv1_2 = util.resize_output(conv1_2, size)
        print 'Concatenating feature maps...'
        feature = np.concatenate((conv5_3, conv4_3, conv3_3, conv2_2, conv1_2))
        """
        
        in_ = np.zeros(size)
        for i in range(20):
            in_ += scoremap[list_index[i],:,:] * list_cos_sim[i]
        first = 0
        scoremap = in_ > (np.mean(in_) * 2 + np.max(in_)) / 3
        
        #print info_[0], info_[1], np.min(in_), np.mean(in_), np.max(in_), 'threshold:', (np.mean(in_) * 2 + np.max(in_)) / 3
        

        # save image
        im = Image.fromarray((scoremap*255).astype(np.uint8), mode='P')
        im_dir = '../data/DAVIS/Predicted_0/480p/{}'.format(info_[0])
        if not os.path.exists(im_dir):
            os.makedirs(im_dir)
        im.save('{}/{:0>5d}.png'.format(im_dir, int(info_[1])))
        
        # motion
        motion = np.array(Image.open('../data/DAVIS/Prior/480p/{}/{:0>5d}.png'.format(info_[0], int(info_[1]))))
        motion = np.asarray([motion])
        motion = util.resize_output(motion, size)
        
        s = generate_binary_structure(2, 2)
        labeled_scoremap, num = label(scoremap, structure=s)
        for i in range(1, num + 1):
            mask = np.zeros(size)
            mask[labeled_scoremap == i] = 1
            
            # remove small segments
            if np.sum(mask) < mask.shape[0] * mask.shape[1] * 1 / 100:
                continue
            
            info_seg.append([info_[0], info_[1], info_[2]])
            
            #start = time.time()
            mask5_3 = util.resize_output(mask, conv5_3.shape)
            mask4_3 = util.resize_output(mask, conv4_3.shape)
            mask3_3 = util.resize_output(mask, conv3_3.shape)
            mask2_2 = util.resize_output(mask, conv2_2.shape)
            mask1_2 = util.resize_output(mask, conv1_2.shape)
            
            feature5_3 = util.get_vector(conv5_3, mask5_3)
            feature4_3 = util.get_vector(conv4_3, mask4_3)
            feature3_3 = util.get_vector(conv3_3, mask3_3)
            feature2_2 = util.get_vector(conv2_2, mask2_2)
            feature1_2 = util.get_vector(conv1_2, mask1_2)
            
            feature_vector = np.concatenate((feature5_3, feature4_3, feature3_3, feature2_2, feature1_2))
            #end = time.time()
            #print end - start
            
            #feature_vector = util.get_vector(feature, mask)
            score_vector = util.get_vector(score, mask)
            motion_vector = util.get_vector(motion, mask)
            #print 'mask:', np.sum(mask), 'score:', score_vector, 'motion:', motion_vector
            
            feature_vectors.append(feature_vector)
            score_vectors.append(score_vector)
            motion_vectors.append(motion_vector)
            masks.append(mask)
            num_segments += 1
        num_images += 1
        print 'Selected images:', num_images, 'Selected segments:', num_segments
        
    score_vectors = np.asarray(score_vectors).astype(float)
    #print 'score:', np.min(score_vectors), np.max(score_vectors)
    score_vectors = (score_vectors - np.min(score_vectors)) / (np.max(score_vectors) - np.min(score_vectors))
    
    motion_vectors = np.asarray(motion_vectors).astype(float)
    #print 'motion:', np.min(motion_vectors), np.max(motion_vectors)
    motion_vectors = (motion_vectors - np.min(motion_vectors)) / (np.max(motion_vectors) - np.min(motion_vectors))
    
    #print 'shape:', np.asarray(feature_vectors).shape, np.asarray(score_vectors).shape, np.asarray(motion_vectors).shape
    return {'info': info_seg, 'feature_vectors': np.asarray(feature_vectors), 'score_vectors': score_vectors, 'motion_vectors': motion_vectors, 'masks': np.asarray(masks)}

def cal_iou(y_pred, y_true):
    y_pred = y_pred.astype('float') / 255
    y_true = y_true.astype('float') / 255
    return np.sum(y_pred * y_true) / np.sum(y_pred + y_true - y_pred * y_true)

def do_submodular_func(video_name):
    """
    with open('../cache/features/features_DAVIS_' + video_name + '.p', 'r') as f:
        feat_dict = cPickle.load(f)
    f.close()
    """
    feat_dict = get_features(video_name)
    
    info = feat_dict['info']
    feature = feat_dict['feature_vectors']
    response = feat_dict['score_vectors']
    motion = feat_dict['motion_vectors']
    mask = feat_dict['masks']
    
    #print 'info', len(info)

    class Opt:
        pass

    opt = Opt()
    opt.gamma = 1
    opt.lambda_fcn = 20
    opt.lambda_motion = 35
    opt.alpha = 1
    
    candi_inds = np.arange(len(feature))
    affmat = util.build_graph(feature)  # pairwise term
    #print 'affmat:', affmat
    
    #start = time.time()
    cur_pos_inds, obj_val = util.submodular_func(affmat, candi_inds, response, motion, opt)
    #end = time.time()
    #print end - start
    
    """    
    iou_list = []
    for i, ind in enumerate(cur_pos_inds):
        info_split = info[ind]
        label_ = Image.open('../data/DAVIS/Annotations/480p/{}/{:0>5d}.png'.format(info_split[0], int(info_split[1])))
        img_ = Image.fromarray(mask[ind] * 255).resize(label_.size)
        img_ = np.array(img_)
        img_[img_>0] = 255
        label_ = np.array(label_)
        label_[label_>0] = 255
        iou = cal_iou(img_, label_) * 100
        
        s = generate_binary_structure(2, 2)
        cc_map, cc_num = label(mask[ind], structure=s)
        
        print len(info), ind, cc_num, np.sum(mask[ind]), 'iou:', iou
        iou_list.append(iou)
    print 'mean iou of {} segments:'.format(video_name), np.nanmean(np.array(iou_list))
    """
        
    info_list = []
    new_masks = []
    for ind in cur_pos_inds:
        if info[ind] not in info_list:
            info_list.append(info[ind])
            new_masks.append(mask[ind])
        else:
            new_masks[info_list.index(info[ind])] += mask[ind]
    
    #iou_list = []
    for i, inf in enumerate(info_list):
        im_dir = '../data/DAVIS/Train/480p/{}'.format(inf[0])
        if not os.path.exists(im_dir):
            os.makedirs(im_dir)
        im = Image.fromarray(new_masks[i].astype(np.uint8) * 255)
        im.save('{}/{:0>5d}.png'.format(im_dir, int(inf[1])))
    """
        label_ = Image.open('../data/DAVIS/Annotations/480p/{}/{:0>5d}.png'.format(inf[0], int(inf[1])))
        img_ = im.resize(label_.size)
        img_ = np.array(img_)
        img_[img_>0] = 255
        label_ = np.array(label_)
        label_[label_>0] = 255
        iou = cal_iou(img_, label_) * 100
        print inf, 'iou', iou
        iou_list.append(iou)
    print 'mean iou of {} images:'.format(video_name), np.nanmean(np.array(iou_list))
    
    print opt.lambda_fcn, opt.lambda_motion
    print len(info_list)
    """
        
    with open('../cache/imagesets/train_{}.txt'.format(video_name), 'w') as ft:
        for info in info_list:
            ft.write(info[0] + ' ' + info[1] + ' ' + str(info[2]) + '\n')
    ft.close()
    
    with open('../cache/imagesets/val_' + video_name + '.txt', 'w') as ft:
        for info in val:
            info_ = info.split()
            if info_[0] == video_name:
                ft.write(info + '\n')
    ft.close()

do_submodular_func(video_name)
