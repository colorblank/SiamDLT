# SiamDLT (Similarity Learning for Dense Label Transfer)
a pytorch implementation of the paper [Similarity Learning for Dense Label Transfer](https://davischallenge.org/challenge2018/papers/DAVIS-Interactive-Challenge-2nd-Team.pdf)



### Example

![boat](result/demo.gif)



### Usage
---

#### requirement:

* pytorch 0.4.0
* python 3.6
* GPU support
* [DAVIS2017 dataset](https://davischallenge.org/davis2017/code.html) 



#### Training(from deeplab pretrained model) 

1. download [deeplab pretrained model](https://drive.google.com/uc?id=1Vi9mFuXk03GBbSV_3smjFA8S5-t3xj1h&export=download) which is support by [pytorch-deeplab-resnet](https://github.com/isht7/pytorch-deeplab-resnet) , put it in pretrained/deeplab.pth
2. modify [DAVIS_PATH](https://github.com/mayorx/SiamDLT/blob/6c6e82a213899e566487bf56909ec34b262cf1ae/dataset.py#L8)
3. run ``` python main.py ```

    

#### Training(from checkpoint)

0. pretrained model will be available soon....
1. modify  [DAVIS_PATH](https://github.com/mayorx/SiamDLT/blob/6c6e82a213899e566487bf56909ec34b262cf1ae/dataset.py#L8)
2. modify [ckpt_file](https://github.com/mayorx/SiamDLT/blob/master/main.py#L21)
3. run ``` python main.py ```

   

#### Inference

1. modify [ckpt_file](https://github.com/mayorx/SiamDLT/blob/master/eval.py#L12)
2. run ``` python eval.py ```
3. results will saved in result/ , you can modify [here](https://github.com/mayorx/SiamDLT/blob/master/utils.py#L54) to see more results

   

### Acknowledgment

The implementation of deeplab is heavily borrowed from [pytorch-deeplab-resnet](https://github.com/isht7/pytorch-deeplab-resnet)



### Reference

* [Similarity Learning for Dense Label Transfer](https://davischallenge.org/challenge2018/papers/DAVIS-Interactive-Challenge-2nd-Team.pdf)

### Citing

If you find this code useful, please cite:
```
@article{DAVIS2018-Interactive-2nd,
          author = {M. Najafi, V. Kulharia, T. Ajanthan, P. H. S. Torr},
          title = {Similarity Learning for Dense Label Transfer},
          journal = {The 2018 DAVIS Challenge on Video Object Segmentation - CVPR Workshops},
          year = {2018}
        }
```
