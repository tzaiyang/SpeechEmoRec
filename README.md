# SpeechEmoRec
[![Building Status](https://travis-ci.org/tzaiyang/SpeechEmoRec.svg?branch=master)](https://travis-ci.org/tzaiyang/SpeechEmoRec)

## Introduction
This project aims to implement speech emotion recognition strategy proposed in *Speech Emotion Recognition Using Deep Convolutional Neural Network and Discriminant Temporal Pyramid Matching*

## Runtime enviorment
*CPU Host :*
+ ubuntu16.04
+ python3.5
+ tensorflow1.7.0

*GPU Server :*
+ tensorflow-gpu1.7.0
+ NVIDIA driver version:390
+ cuda9.0
+ cudnn7.0

## Instructions
1. Loading Berlin Database of Emotional Speech!
> python load_emodb.py
2. Preprocessing Data
> python melSpec.py
3. Finetune AlexNet with Tensorflow
> python finetune.py

## Refrences:
Refrence Model:
+ Alexnet

Refrence Papers:
+ *ImageNet Classification with Deep Convolutional*
Neural Networks
+ *Speech Emotion Recognition Using Deep Convolutional Neural Network and Discriminant Temporal Pyramid Matching*
+ *Geometric ℓp-norm feature pooling for image classification*
