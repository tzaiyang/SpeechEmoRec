from __future__ import division, print_function, absolute_import
import numpy as np,wave
import scipy as sp
import matplotlib.pyplot as plt
import PIL.Image as Image
import os,sys
import librosa
import librosa.display
from pylab import *

# if you want to see utterance mel spectrogram and delta,delta-delta picture,
# set __DEBUG_ as True,and the pictures will be DEBUG directory

def gen_filename_txt(Dataset_EMODB):
    fl=open('%s/../emodbname.txt'%(Dataset_EMODB),'w')
    wave_filenames = os.listdir(Dataset_EMODB)
    for filename in wave_filenames:
        fl.write(filename)
        fl.write('  ')
        fl.write(filename[5:6])
        fl.write('\n')
    fl.close

def read_wav(wav_path):
    wavefile = wave.open(wav_path)
    nchannels,sampwidth,framerate,nframes,comptype,compname=wavefile.getparams()
    strdata = wavefile.readframes(nframes)
    wavedata = np.fromstring(strdata, dtype=np.int16).astype('float32')# / (2 ** 15)

    print ('nchnnels:%d'%nchannels)
    print ('sampwidth:%d'%sampwidth)
    print ('framerate:%d'%framerate)
    print ('nframes:%d'%nframes)
    #print (wavedata)
    wavefile.close()
    return nchannels,sampwidth,framerate,nframes,wavedata

def gen_utterance_melspec(wav_path):
    """
    Compute a mel-scaled spectrogram to a utterance wavefile
    :param wav_path: audio time-series file
    :return:
    """
    nchannels,sampwidth,framerate,nframes,wavedata = read_wav(wav_path)
    #wavedata,framerate = librosa.core.load(wav_path)

    # method 1:hamming window
    # sp.signal.get_window('hamming', 7)
    # Zxx = librosa.core.stft(wavedata, n_fft=25*framerate/1000, hop_length=(25-10)*framerate/1000, window='hamming', center=True, pad_mode='reflect')
    # Sxx = librosa.feature.melspectrogram(S=np.abs(Zxx),n_mels=64, fmin=20, fmax=8000)
    # method 2:hanning window
    Sxx = librosa.feature.melspectrogram(y=wavedata,sr=framerate,n_fft=(int)(25*framerate/1000),hop_length=(int)((10)*framerate/1000),n_mels=64,fmin=20,fmax=8000)
    #librosa.time_to_frames()
    print(Sxx.shape)
    return Sxx

def save_utterance(X,savepath,filename="melSpec"):
    # Convert a power spectrogram (amplitude squared) to decibel (dB) units
    X = librosa.power_to_db(X, ref=np.max)
    # Display a spectrogram/chromagram/cqt/etc.
    librosa.display.specshow(X,fmin=20, fmax=8000)
    plt.savefig("%s/%s"%(savepath,filename),bbox_inches='tight',pad_inches=0)
    close()

def gen_segments_melspec(X, window_size, overlap_sz):
    """
    Create an overlapped version of X

    Parameters
    ----------
    X : ndarray, shape=(n_mels,n_samples)
        Input signal to window and overlap

    window_size : int
        Size of windows to take

    overlap_sz : int
        Step size between windows

    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    window_step = (window_size-overlap_sz)
    append = np.zeros((64,(window_step - (X.shape[-1]-overlap_sz) % window_step)))
    X = np.hstack((X, append))
    # append zeros of end of X to get integer numbers of n_windows
    print(X.shape)
    new_shape = ((X.shape[-1] - overlap_sz) // window_step,window_size,X.shape[0])
    new_strides = (window_step*8,X.strides[0],X.strides[-1])
    X_strided = np.lib.stride_tricks.as_strided(X, shape=new_shape, strides=new_strides)

    return X_strided

def normlize(x):
    return ((x-np.min(x))/(np.max(x)-np.min(x)))

def save_segment(X,pic_path):
    # librosa.display.specshow(X, fmin=20, fmax=8000)
    # plt.savefig(pic_path, bbox_inches='tight', pad_inches=0)
    plt.imsave(pic_path,X)
    #sp.misc.imsave(pic_path, X)
    close()

def save_dcnn_ipput(X,pic_path):
    #plt.imsave(pic_path,X)
    sp.misc.imsave(pic_path,X)
    #sp.misc.toimage(X).save(pic_path)
    close()
    # pri_image = Image.open(pic_path)
    # pri_image.resize((227,227),Image.ANTIALIAS).save(pic_path)


def gen_dcnn_input(wav_path,savepath,filename):
    utterance_melspec = gen_utterance_melspec(wav_path)
    if _DEBUG_:
        save_utterance(utterance_melspec,savepath)

    segments_melspec = gen_segments_melspec(utterance_melspec,window_size=64,overlap_sz=64-30)
    train_file = open(TrainDataset_FILENAME,'a')
    val_file = open(ValDataset_FILENAME,'a')

    utterance_train_file = open(TrainDataset_FILENAMES,'a')
    utterance_val_file = open(ValDataset_FILENAMES,'a')
    if not _DEBUG_:
        if wav_path.split('/')[-1][:2] == '09':
            utterance_val_file.write('%s %s\n' % (savepath, labels_dict[filename]))
        else:
            utterance_train_file.write('%s %s\n' % (savepath, labels_dict[filename]))
    for num in range(0, segments_melspec.shape[0]):
        static = librosa.power_to_db(segments_melspec[num], ref=np.max)
        delta = librosa.feature.delta(static, order=1)
        delta2 = librosa.feature.delta(static, order=2)

        static = normlize(static)*255
        delta = normlize(delta)*255
        delta2 = normlize(delta2)*255

        if _DEBUG_:
            save_segment(static,pic_path="%s/%s%d.png" % (savepath, "static", num))
            save_segment(delta, pic_path="%s/%s%d.png" % (savepath, "delta", num))
            save_segment(delta2,pic_path = "%s/%s%d.png" % (savepath, "delta_", num))

        images = np.dstack((static,delta,delta2))

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        pic_path = '%s/%s%d.png' % (savepath, filename, num)
        if not _DEBUG_:
            # configuring '09' speaker as validation dataset
            if wav_path.split('/')[-1][:2] == '09':
                val_file.write('%s %s\n'%(pic_path,labels_dict[filename]))
            else:
                train_file.write('%s %s\n'%(pic_path,labels_dict[filename]))
        #dcnninname.write('%s %s\n' % (pic_path, labels_dict[filename]))
        save_dcnn_ipput(images,pic_path)
    train_file.close()
    val_file.close()
    utterance_train_file.close()
    utterance_val_file.close()

if __name__ == '__main__':
    Dataset_EMODB = "Dataset/EMODB"
    Dataset_DCNN = "Dataset/DCNN_IN"
    Dataset_DEBUG = "Dataset/DEBUG"
    TrainDataset_FILENAME = "Dataset/train_file.txt"
    ValDataset_FILENAME = "Dataset/val_file.txt"
    TrainDataset_FILENAMES = "Dataset/train.txt"
    ValDataset_FILENAMES = "Dataset/val.txt"
    #
    # labels_dict = {'W':[1,0,0,0,0,0,0],
    #                'L':[0,1,0,0,0,0,0],
    #                'E':[0,0,1,0,0,0,0],
    #                'A':[0,0,0,1,0,0,0],
    #                'F':[0,0,0,0,1,0,0],
    #                'T':[0,0,0,0,0,1,0],
    #                'N':[0,0,0,0,0,0,1]}

    labels_dict = {'W':0,
                   'L':1,
                   'E':2,
                   'A':3,
                   'F':4,
                   'T':5,
                   'N':6
    }

    if len(sys.argv) > 1:
        if sys.argv[1] == '-d':
            _DEBUG_ = True
    else:
        _DEBUG_ = False

    gen_filename_txt(Dataset_EMODB)
    if not _DEBUG_:
        if os.path.exists(TrainDataset_FILENAME):
            os.remove(TrainDataset_FILENAME)
        if os.path.exists(ValDataset_FILENAME):
            os.remove(ValDataset_FILENAME)
    wave_filenames = os.listdir(Dataset_EMODB)
    for filename in wave_filenames:
        wav_path = '%s/%s' % (Dataset_EMODB, filename)
        save_path = '%s/%s' % (Dataset_DCNN, filename)
        if _DEBUG_:
            if not os.path.exists(Dataset_DEBUG):
                os.makedirs(Dataset_DEBUG)
            gen_dcnn_input(wav_path, Dataset_DEBUG, filename[5:6])
            print(filename)
            break
        gen_dcnn_input(wav_path, save_path, filename[5:6])

