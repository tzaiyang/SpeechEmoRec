# SpeechEmoRec
Speech Emotion Recognition Using Deep Convolutional Neural Network and Discriminant Temporal Pyramid Matching

## 运行环境要求
ubuntu16.04, python3.5, tensorflow1.4.1, opencv3.4.0

## 代码运行说明
### 运行准备工作
+ 在工作目录下建立AlexForAudio_In_Out文件夹作为数据主目录
+ 在AlexForAudio_In_Out中放入数据预处理代码的输出结果，即melSpec_Bolin_Speech，和txts两个文件夹
+ 在global_path.py中更改输入输出数据的主目录路徑:如
Data_Directory = '/home/ryan/Documents/AlexForAudio_In_Out/'


### 运行步骤
+ 运行train_alexnet.py训练alexnet网络
+ 运行get_train_features.py得到训练集全局特征
+ 运行get_test_features.py得到测试集全局特征
+ 运行train_svm.py训练svm,至此可以得到整个学习的模型结果
+ 运行classifier.py用得到的模型进行语音情感识别
