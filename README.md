# TransferSeg

Caffe implementation of our method for transferring knowledge from seen objects in images to unseen objects in videos. <br />
Contact: Yi-Wen Chen (chenyiwena at gmail dot com)

<p align="center">
<img src="https://github.com/wenz116/TransferSeg/blob/master/figure/overview.png" width="75%">
</p>

## Paper

Please cite our paper if you find it useful for your research.

[Unseen Object Segmentation in Videos via Transferable Representations](https://arxiv.org/abs/1901.02444) <br />
[Yi-Wen Chen](https://wenz116.github.io/), [Yi-Hsuan Tsai](https://sites.google.com/site/yihsuantsai/home), Chu-Ya Yang, [Yen-Yu Lin](https://www.citi.sinica.edu.tw/pages/yylin/) and [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/index.html) <br />
*Asian Conference on Computer Vision (ACCV)*, 2018 (**oral**) <br />
**Best Student Paper Award Honorable Mention**

```
@inproceedings{Chen_TransferSeg_2018,
  author = {Yi-Wen Chen and Yi-Hsuan Tsai and Chu-Ya Yang and Yen-Yu Lin and Ming-Hsuan Yang},
  booktitle = {Asian Conference on Computer Vision (ACCV)},
  title = {Unseen Object Segmentation in Videos via Transferable Representations},
  year = {2018}
}
```

[VOSTR: Video Object Segmentation via Transferable Representations](https://wenz116.github.io/files/IJCV20_VOSTR.pdf) <br />
[Yi-Wen Chen](https://wenz116.github.io/), [Yi-Hsuan Tsai](https://sites.google.com/site/yihsuantsai/home), [Yen-Yu Lin](https://www.citi.sinica.edu.tw/pages/yylin/) and [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/index.html) <br />
*International Journal of Computer Vision (IJCV)*, 2020 <br />

```
@inproceedings{Chen_VOSTR_2020,
  author = {Yi-Wen Chen and Yi-Hsuan Tsai and Yen-Yu Lin and Ming-Hsuan Yang},
  journal = {International Journal of Computer Vision (IJCV)},
  title = {VOSTR: Video Object Segmentation via Transferable Representations},
  volume = {128},
  number = {4},
  pages = {931-949},
  year = {2020}
}
```

## Installation
* Install Caffe: http://caffe.berkeleyvision.org/.

* Install MATLAB

* Clone this repo
```
git clone https://github.com/wenz116/TransferSeg.git
cd TransferSeg
```

* Prepare for MBS

1. Go to the folder `utils/MBS/mex`.

2. Modify the opencv include and lib paths in `compile.m/compile_win.m` (for Linux/Windows).

3. Run `compile/compile_win` in MATLAB (for Linux/Windows).

## Dataset
* Download the [PASCAL VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/index.html) as the source image dataset, and put it in the `data/PASCAL/VOC2011` folder.

* Download the [DAVIS Dataset](https://davischallenge.org/index.html) as the target video dataset, and put it in the `data/DAVIS` folder.

## Training
* Download the [FCN model](http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel) pre-trained on PASCAL VOC, and put it in the `nets` folder.

* Go to the folder `scripts`.

1. Compute optical flow of the input video. Run `compute_optical_flow('<VIDEO_NAME>')` in MATLAB. The optical flow images will be saved at `data/DAVIS/Motion/480p/<VIDEO_NAME>/`.

2. Compute motion prior of the input video via minimum barrier distance. Run `get_prior('<VIDEO_NAME>')` in MATLAB. The motion prior images will be saved at `data/DAVIS/Prior/480p/<VIDEO_NAME>/`.

3. Extract features of each category in PASCAL VOC. The extracted features will be saved at `cache/features/`, named as `features_PASCAL_<CLASS_NAME>_fc7.p`.
```
python get_feature_PASCAL.py <GPU_ID>
```

4. Extract features of the input video. The extracted features will be saved at `cache/features/`, named as `features_DAVIS_<VIDEO_NAME>_fc7.p`.
```
python get_feature_DAVIS.py <GPU_ID> <VIDEO_NAME>
```

5. Segment mining. The selected segments will be saved at `data/DAVIS/Train/480p/<VIDEO_NAME>/`.
```
python get_score.py <GPU_ID> <VIDEO_NAME>
```

6. Self learning. The trained models will be saved at `output/snapshot/`.
```
./train.sh <GPU_ID> <VIDEO_NAME>
```

## Note
The model and code are available for non-commercial research purposes only.
* 12/2018: code released
