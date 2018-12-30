import numpy as np
import scipy

def get_output(net, blob_name):
    return net.blobs[blob_name].data[0]

def get_output_score(net):
    return net.blobs['score'].data[0].max(axis=0)

def get_out_scoremap(net):
    return net.blobs['score'].data[0].argmax(axis=0).astype(np.uint8)

def feed_net(net, in_):
    """
    Load prepared input into net.
    """
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_

def segrun(net, in_):
    feed_net(net, in_)
    net.forward()
    return get_output(net, 'score'), get_output_score(net), get_output(net, 'conv5_3'), get_output(net, 'conv4_3'), get_output(net, 'conv3_3'), get_output(net, 'conv2_2'), get_output(net, 'conv1_2')

def segrun_fc7(net, in_):
    feed_net(net, in_)
    net.forward()
    return get_out_scoremap(net), get_output(net, 'fc7')

def resize_output(x, size):
    if len(x.shape) == 3 and len(size) == 2:
        return scipy.ndimage.zoom(x, zoom=(1, float(size[0]) / float(x.shape[1]), float(size[1]) / float(x.shape[2])), order=3)
    elif len(x.shape) == 2 and len(size) == 3:
        return scipy.ndimage.zoom(x, zoom=(float(size[1]) / float(x.shape[0]), float(size[2]) / float(x.shape[1])), order=3)
    else:
        return scipy.ndimage.zoom(x, zoom=(float(size[0]) / float(x.shape[0]), float(size[1]) / float(x.shape[1])), order=2)

def get_vector(array, mask):
    masked_array = np.multiply(array, mask)
    num_pixels = float(np.sum(mask))
    vector = np.zeros(array.shape[0])
    for i in range(masked_array.shape[0]):
        vector[i] = np.sum(masked_array[i]) / num_pixels
    return vector

def edges_between(inds):
    num = len(inds)
    assert num > 0
    mat = np.tril(np.ones((num, num)), -1)
    return np.argwhere(mat == 1.0)

def gene_distance_feature(feat1, feat2):
    valDistances = np.sum((feat1 * feat2), axis=1)
    minVal = np.amin(valDistances)
    valDistances = (valDistances - minVal) / (np.amax(valDistances) - minVal)
    return valDistances

def gene_sub_weight(row, col, feature):
    return gene_distance_feature(feature[row, :], feature[col, :])

def gene_sub_graph(weights, row, col, indNum):
    affmat = np.zeros((indNum, indNum))
    affmat[row, col] = weights
    affmat[col, row] = weights
    return affmat

def build_graph(feature):
    indNum = len(feature)
    inds = np.arange(indNum)
    edges = edges_between(inds)
    row = edges[:, 0]
    col = edges[:, 1]
    weights = gene_sub_weight(row, col, feature)
    return gene_sub_graph(weights, row, col, indNum)

def gene_obj_val(affmat, pos_label_inds, response, motion, opt):
    lambda_fcn = opt.lambda_fcn
    lambda_motion = opt.lambda_motion
    alpha = opt.alpha
    gamma = opt.gamma
    return alpha * np.sum(affmat[pos_label_inds, :]) - np.sum(gamma * len(pos_label_inds)) + lambda_fcn * np.sum(response) + lambda_motion * np.sum(motion)

def submodular_func(affmat, candi_inds_all, response, motion, opt):
    cur_pos_inds = []
    obj_val = []

    if len(candi_inds_all) <= 2:
        seedNum = 1
    else:
        seedNum = np.floor(len(candi_inds_all) * 0.8).astype(int)
    #print seedNum
    for i in range(seedNum):
        s = set(cur_pos_inds)
        cand_pos_label_inds = [x for x in candi_inds_all if x not in s]
        num_cand = len(cand_pos_label_inds)
        cand_ranks = np.zeros((num_cand, 1))
        cand_obj_val = np.zeros((num_cand, 1))
        for j in range(num_cand):
            pos_label_inds = cur_pos_inds + [cand_pos_label_inds[j]]
            cand_obj_val[j] = gene_obj_val(affmat, pos_label_inds, response[pos_label_inds], motion[pos_label_inds], opt)
    
        max_val = np.amax(cand_obj_val)
        max_ind = np.argmax(cand_obj_val)
        
        #if i == 0:
        #    print i, max_val
        #elif i == 1:
        #    print i, max_val, 'diff:', max_val - obj_val[i - 1]
        #else:
        #    print i, max_val, 'diff:', max_val - obj_val[i - 1], 'ratio:', (max_val - obj_val[i - 1]) / (obj_val[i - 1] - obj_val[i - 2])
        
        if i > 0 and max_val < obj_val[i - 1]:
            #print '----break 1----'
            break
        if i > 1 and (max_val - obj_val[i - 1]) < (obj_val[i - 1] - obj_val[i - 2]) * 0.8:
            #print '----break 2----'
            break
        
        cur_pos_inds.append(cand_pos_label_inds[max_ind])
        obj_val.append(max_val)

    return cur_pos_inds, obj_val
