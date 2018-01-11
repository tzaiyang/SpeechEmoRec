#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np, wave

np.set_printoptions(threshold=np.inf)

from global_path import Data_Directory

test_path = '%s%s' % (Data_Directory, 'test.wav')


def getSpectrum(fileStr, window_length_ms=25,windowOver_lenth_ms=10,frameinseg_lenth_num=64,frameinsegOver_lenth_num=30):
    # window_length_ms  是以毫秒为单位的窗长
    # frame_shift_times 是以毫秒为单位的帧移
    # frame_length_ms   是以毫秒为单位的帧长
        # 读音频文件
        wav_file = wave.open(fileStr, 'r')
        # 获取音频文件的各种参数
        params = wav_file.getparams()
        nchannels, sampwidth, framerate, wav_length = params[:4]
        # 获取音频文件内的数据，不知道为啥获取到的竟然是个字符串，还需要在numpy中转换成short类型的数据
        str_data = wav_file.readframes(wav_length)
        wave_data = np.fromstring(str_data, dtype=np.short)
        s = len(wave_data)

        # 将窗长从毫秒转换为点数
        window_length = int(framerate * window_length_ms / 1000)
        windowOver_lenth = int(framerate * windowOver_lenth_ms / 1000)
        windowOver_lenth = int( window_length / 2)
        frameinseg_lenth_ms = 10*(frameinseg_lenth_num-1)+25
        frameinsegOver_lenth_ms = 10 * (frameinsegOver_lenth_num - 1) + 25
        frameinseg_length = int(framerate * frameinseg_lenth_ms / 1000)
        frameinsegOver_length = int(framerate * frameinsegOver_lenth_ms / 1000)
        # print("frame_length:", frameinseg_length)
        # print("frameinsegOver_length:", frameinsegOver_length)
        ##初始化窗口

        h=window_length-windowOver_lenth
        win = np.hamming(window_length)

        # 计算总帧数，并创建一个空矩阵
        # nframe =int(wav_length / frame_length)
        # 循环计算每一个窗内的fft值
        startN = 0
        c=1
        # filePatholdd = genDir(filePathNames, "old")
        # filePathneww = genDir(filePathNames, "new")
        images = []
        while ((s-startN)>=frameinseg_length):
            ncols = 1 + int((frameinseg_length - window_length) / h)
            spec = np.zeros((int(window_length / 2), ncols))

            for i in range(0,ncols):
                start = startN+i*h
                end = start + window_length  # [:window_length/2]是指只留下前一半的fft分量
                u = [x * y for x, y in zip(win, wave_data[start:end])]
                # print(type(u))
                # print(i)
                spec[:, i] = np.log(np.abs(np.fft.fft(u)))[:int(window_length / 2)]

            c = c + 1
            startN=startN+frameinseg_length-frameinsegOver_length
            images.append(spec)

        return images


# 测试代码
if __name__ == '__main__':
    spe = getSpectrum(test_path)
    print(len(spe))
    for i in range(len(spe)):
        print(spe[i].shape)
