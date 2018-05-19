SpeechEmoRec
============

[![Building Status](https://travis-ci.org/tzaiyang/SpeechEmoRec.svg?branch=master)](https://travis-ci.org/tzaiyang/SpeechEmoRec)

Introduction
------------
This project aims to implement speech emotion recognition strategy proposed in *Speech Emotion Recognition Using Deep Convolutional Neural Network and Discriminant Temporal Pyramid Matching*

Runtime enviorment
------------------
*CPU Host :* 
+ ubuntu16.04 
+ python3.5
+ tensorflow1.7.0

*GPU Server :*
+ tensorflow-gpu1.7.0
+ NVIDIA driver version:390
+ cuda9.0
+ cudnn7.0

Instructions
------------
### Preprocessing Data
1. Update path of dataset which you want to save from path.py   
2. Downloading Berlin Database of Emotional Speech!  
   1. *Berlin Dataset*  
    ```$ python load_emodb.py```
   2. *eNTERFACE Dataset*  
     Downloading the eNTERFACE05 Dataset and update the dataset root
3. Starting preprocessing 
    
    $ python melSpec.py

### Feature Extracting 
Finetune AlexNet with Tensorflow

    $ python finetune.py
Discriminant Temporal Pyramid Matching

    $ python dtpm.py -s  
    $ python dtpm.py -n

### Classfier
Support Vector Machine

    $ python svm.py

Refrences:
---------
Refrence Model:
+ Alexnet
+ SVM

Refrence Papers:
+ *ImageNet Classification with Deep Convolutional*
Neural Networks
+ *Speech Emotion Recognition Using Deep Convolutional Neural Network and Discriminant Temporal Pyramid Matching*
+ *Geometric ℓp-norm feature pooling for image classification*
