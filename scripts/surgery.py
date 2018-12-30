from __future__ import division
import caffe
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import cPickle

def transplant(new_net, net, suffix=''):
    """
    Transfer weights by copying matching parameters, coercing parameters of
    incompatible shape, and dropping unmatched parameters.

    The coercion is useful to convert fully connected layers to their
    equivalent convolutional layers, since the weights are the same and only
    the shapes are different.  In particular, equivalent fully connected and
    convolution layers have shapes O x I and O x I x H x W respectively for O
    outputs channels, I input channels, H kernel height, and W kernel width.

    Both  `net` to `new_net` arguments must be instantiated `caffe.Net`s.
    """
    for p in net.params:
        p_new = p + suffix
        if p_new not in new_net.params:
            print 'dropping', p
            continue
        for i in range(len(net.params[p])):
            if i > (len(new_net.params[p_new]) - 1):
                print 'dropping', p, i
                break
            if net.params[p][i].data.shape != new_net.params[p_new][i].data.shape:
                print 'coercing', p, i, 'from', net.params[p][i].data.shape, 'to', new_net.params[p_new][i].data.shape
            else:
                print 'copying', p, ' -> ', p_new, i
            new_net.params[p_new][i].data.flat = net.params[p][i].data.flat

def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

def interp(net, layers):
    """
    Set weights of each layer in layers to bilinear kernels for interpolation.
    """
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k and k != 1:
            print 'input + output channels need to be the same or |output| == 1'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt

def expand_score(new_net, new_layer, net, layer):
    """
    Transplant an old score layer's parameters, with k < k' classes, into a new
    score layer with k classes s.t. the first k' are the old classes.
    """
    old_cl = net.params[layer][0].num
    new_net.params[new_layer][0].data[:old_cl][...] = net.params[layer][0].data
    new_net.params[new_layer][1].data[0,0,0,:old_cl][...] = net.params[layer][1].data

def init_score_weight(net, video_name):
    """
    Initialize the weight of the 1*1 convolutional layer as the perceptron score.
    """
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
    
    for i in range(20):
        net.params['conv_weight'][0].data[:,list_index[i],:,:] = list_cos_sim[i]
        #print video_name, pv_classes[list_index[i]], list_cos_sim[i]
    #print net.params['conv_weight'][0].data[0]
