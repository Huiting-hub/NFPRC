# NFPRC

This is the code for: Noise-Free Prototype Guided Representation Calibration under Label Noise (PyTorch implementation).

## Dependencies

- Ubuntu 22.04

- Python 3.9

- PyTorch, verion=2.0.0

- CUDA, version=11.8

## Experiments

We verify the effectiveness of the proposed method on synthetic noisy datasets. In this repository, we provide the used [datasets](https://drive.google.com/open?id=1Tz3W3JVYv2nu-mdM6x33KSnRIY1B7ygQ) (the images and labels have been processed to .npy format). You should put the datasets in the folder “data” when you have downloaded them.
Training example:

```
python main.py --dataset mnist --noise_type symmetric --noise_rate 0.4 --Lambda1 0.03 --Lambda2 0.01 --seed 1
```

## Pretrained Models

Our pre-trained models by unsupervised contrastive learning for computing prototypes can be downloaded as following: [Models](https://drive.google.com/drive/folders/1TECIuxCObR_UEIYN1qsItLl9CvuwzUBx?hl=zh-cn)

You can also pretrain your own models by:

```
python moco_v2.py --learning_rate 0.06 --batch_size 256 
```

## BibTeX

```
@article{YUAN2025114308,
title = {Noise-free prototype guided representation calibration under label noise},
journal = {Knowledge-Based Systems},
volume = {329},
pages = {114308},
year = {2025},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2025.114308},
}
```

## Acknowledgement

We referred to the official code of [MoCo](https://github.com/facebookresearch/moco).


