# NFPRC

This is the code for: Tackling Noisy Labels with Noise-Free Prototype Guided Representation Calibration (PyTorch implementation).

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
python moco_v2.py --learning_rate 0.01 --batch_size 128 
```
## Acknowledgement

We referred to the official code of [MoCo](https://github.com/facebookresearch/moco).
