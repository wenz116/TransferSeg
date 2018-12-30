import sys
import caffe
import surgery, score

import numpy as np
import matplotlib.pyplot as plt
import os

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

import scipy
import cPickle

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageFilter

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

video_name = sys.argv[2]

weights = '../nets/fcn8s-heavy-pascal.caffemodel'

solver = caffe.SGDSolver('../nets/solver_' + video_name + '.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)
surgery.init_score_weight(solver.net, video_name)

with open('../cache/imagesets/val_' + video_name + '.txt', 'r') as f:
    val = f.read().splitlines()
f.close()

iou_list = []
iou, iou_all_before = score.seg_tests(solver, 1, val, layer='score_weight')

iou_list.append(iou * 100)
iou_best = iou
iteration_best = 0
step_size = 1000
step_num = 5
iter_num = step_size * step_num
for _ in range(step_num):
    solver.step(step_size)
    iou, iou_all_after = score.seg_tests(solver, 2, val, layer='score_weight')
    iou_list.append(iou * 100)
    if iou > iou_best:
        iou_best = iou
        iteration_best = (_ + 1) * step_size
#print solver.net.params['conv_weight'][0].data[0]
print 'The best IoU of ' + video_name + ' is ' + str(iou_best * 100) + '% when iteration = ' + str(iteration_best)
print iou_list

with open('../output/result/result.txt', 'a') as ft:
    ft.write('The best IoU of ' + video_name + ' is ' + str(iou_best * 100) + '% when iteration = ' + str(iteration_best) + '\n')
    ft.write(str(iou_list) + '\n\n')
ft.close()

with open('../output/result/iou_' + video_name + '.p', 'w') as f:
    cPickle.dump((iou_all_before, iou_all_after), f)

plt.figure()
plt.plot(np.arange(0, iter_num+1, step_size), np.array(iou_list), 'bo', np.arange(0, iter_num+1, step_size), np.array(iou_list), 'k')
plt.axis([0, iter_num, 0, 100])
plt.xlabel('iteration')
plt.ylabel('IoU of ' + video_name + ' (%)')
plt.savefig('../output/figure/' + video_name + '.png')
